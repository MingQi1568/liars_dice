"""Tune Smart bot parameters and try variants. Fast version."""
import random
from collections import Counter
from game import Bot, Bid, GameState
from bots import (
    RandomBot, NaiveBot, ProbabilisticBot, AggressiveBot, ConservativeBot, SmartBot,
    next_bid_options, count_face_in_hand, prob_bid_valid, binom_at_least,
)
from simulate import tournament, print_results, head_to_head, run_game


class SmartBotV2(Bot):
    """Tunable parameters."""
    name = "SmartV2"

    def __init__(self,
                 base_call_threshold: float = 0.30,
                 high_ratio_threshold: float = 0.45,
                 ratio_breakpoint: float = 0.6,
                 bid_threshold: float = 0.55,
                 hand_bonus_weight: float = 0.05,
                 qty_penalty: float = 0.02,
                 bluff_prob_floor: float = 0.5):
        self.base_call_threshold = base_call_threshold
        self.high_ratio_threshold = high_ratio_threshold
        self.ratio_breakpoint = ratio_breakpoint
        self.bid_threshold = bid_threshold
        self.hand_bonus_weight = hand_bonus_weight
        self.qty_penalty = qty_penalty
        self.bluff_prob_floor = bluff_prob_floor

    def act(self, state: GameState):
        my_dice = state.my_dice
        if state.current_bid is not None:
            cb = state.current_bid
            p_valid = prob_bid_valid(state, cb)
            ratio = cb.quantity / max(1, state.total_dice)
            threshold = self.base_call_threshold
            if ratio > self.ratio_breakpoint:
                threshold = self.high_ratio_threshold
            if p_valid < threshold:
                return "liar"

        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice + 1]

        scored = []
        for b in options:
            p = prob_bid_valid(state, b)
            in_hand = count_face_in_hand(my_dice, b.face)
            hand_bonus = self.hand_bonus_weight * in_hand
            qty_penalty = self.qty_penalty * (b.quantity - (state.current_bid.quantity if state.current_bid else 0))
            score = p + hand_bonus - qty_penalty
            scored.append((score, p, b))

        decent = [(s, p, b) for (s, p, b) in scored if p >= self.bid_threshold]
        if decent:
            decent.sort(reverse=True)
            return decent[0][2]

        if state.current_bid is not None:
            p_current = prob_bid_valid(state, state.current_bid)
            if p_current >= self.bluff_prob_floor:
                scored.sort(reverse=True)
                return scored[0][2]
            return "liar"

        scored.sort(reverse=True)
        return scored[0][2]


class SignalBot(Bot):
    """Bayesian: opponent bids likely reflect their dice."""
    name = "Signal"

    def __init__(self, call_threshold=0.32, bid_threshold=0.55, signal_boost=0.04):
        self.call_threshold = call_threshold
        self.bid_threshold = bid_threshold
        self.signal_boost = signal_boost

    def signal_adjusted_prob(self, state: GameState, bid: Bid) -> float:
        in_hand = count_face_in_hand(state.my_dice, bid.face)
        hidden = state.total_dice - len(state.my_dice)
        need = bid.quantity - in_hand
        p_each = 1/6 if bid.face == 1 else 2/6
        # Boost slightly if opponents have bid this face (info that someone holds it)
        signal_count = sum(1 for bidder, b in state.bid_history if bidder != state.my_index and b.face == bid.face)
        adj = min(0.15, self.signal_boost * signal_count)
        p_adj = min(0.99, p_each + adj)
        return binom_at_least(hidden, need, p_adj)

    def act(self, state: GameState):
        if state.current_bid is not None:
            p = self.signal_adjusted_prob(state, state.current_bid)
            ratio = state.current_bid.quantity / max(1, state.total_dice)
            threshold = self.call_threshold + (0.10 if ratio > 0.6 else 0)
            if p < threshold:
                return "liar"

        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice + 1]
        scored = []
        for b in options:
            p = self.signal_adjusted_prob(state, b)
            in_hand = count_face_in_hand(state.my_dice, b.face)
            score = p + 0.05 * in_hand - 0.02 * (b.quantity - (state.current_bid.quantity if state.current_bid else 0))
            scored.append((score, p, b))
        decent = [(s, p, b) for (s, p, b) in scored if p >= self.bid_threshold]
        if decent:
            decent.sort(reverse=True)
            return decent[0][2]
        if state.current_bid is not None:
            p_current = self.signal_adjusted_prob(state, state.current_bid)
            if p_current >= 0.5:
                scored.sort(reverse=True)
                return scored[0][2]
            return "liar"
        scored.sort(reverse=True)
        return scored[0][2]


def eval_variant(variant_factory, opponents_factory_list, games=300, players_per_game=4):
    """Test a variant in fixed-opponent games. Returns win rate."""
    wins = 0
    for _ in range(games):
        bots = [variant_factory()] + [random.choice(opponents_factory_list)() for _ in range(players_per_game - 1)]
        labels = ["VARIANT"] + [f"OPP{i}" for i in range(players_per_game - 1)]
        winner, _ = run_game(bots, labels=labels)
        if winner == "VARIANT":
            wins += 1
    return wins / games


def grid_search_smart():
    """Tune SmartV2 thresholds."""
    opponents = [
        lambda: SmartBot(),
        lambda: AggressiveBot(),
        lambda: ProbabilisticBot(),
        lambda: ConservativeBot(),
    ]
    print("Grid searching SmartV2 (300 games each, 4-player against mixed opponents):")
    print(f"{'ct':>5} {'bt':>5} {'rbp':>5} {'win%':>7}")
    results = []
    for ct in [0.25, 0.30, 0.35]:
        for bt in [0.45, 0.55, 0.65]:
            for rbp in [0.5, 0.7]:
                wr = eval_variant(
                    lambda ct=ct, bt=bt, rbp=rbp: SmartBotV2(base_call_threshold=ct, bid_threshold=bt, ratio_breakpoint=rbp),
                    opponents, games=300,
                )
                results.append((wr, ct, bt, rbp))
                print(f"{ct:>5} {bt:>5} {rbp:>5} {wr*100:>6.2f}%")
    results.sort(reverse=True)
    return results


def grid_search_signal():
    opponents = [
        lambda: SmartBot(),
        lambda: AggressiveBot(),
        lambda: ProbabilisticBot(),
        lambda: ConservativeBot(),
    ]
    print("\nGrid searching SignalBot (300 games each):")
    print(f"{'ct':>5} {'bt':>5} {'sb':>5} {'win%':>7}")
    results = []
    for ct in [0.28, 0.33, 0.38]:
        for bt in [0.50, 0.55, 0.60]:
            for sb in [0.02, 0.04, 0.08]:
                wr = eval_variant(
                    lambda ct=ct, bt=bt, sb=sb: SignalBot(call_threshold=ct, bid_threshold=bt, signal_boost=sb),
                    opponents, games=300,
                )
                results.append((wr, ct, bt, sb))
                print(f"{ct:>5} {bt:>5} {sb:>5} {wr*100:>6.2f}%")
    results.sort(reverse=True)
    return results


if __name__ == "__main__":
    random.seed(7)
    print("=" * 60)
    smart_results = grid_search_smart()
    print(f"\nTop 3 SmartV2 configs:")
    for wr, ct, bt, rbp in smart_results[:3]:
        print(f"  ct={ct}, bt={bt}, rbp={rbp} -> {wr*100:.2f}%")

    print("\n" + "=" * 60)
    signal_results = grid_search_signal()
    print(f"\nTop 3 Signal configs:")
    for wr, ct, bt, sb in signal_results[:3]:
        print(f"  ct={ct}, bt={bt}, sb={sb} -> {wr*100:.2f}%")

    print("\n" + "=" * 60)
    print("\n=== Final tournament with best variants ===")
    best_smart = smart_results[0]
    best_signal = signal_results[0]
    factories = {
        "Smart_orig": lambda: SmartBot(),
        "SmartV2_tuned": lambda bs=best_smart: SmartBotV2(base_call_threshold=bs[1], bid_threshold=bs[2], ratio_breakpoint=bs[3]),
        "Signal_tuned": lambda bs=best_signal: SignalBot(call_threshold=bs[1], bid_threshold=bs[2], signal_boost=bs[3]),
        "Aggressive": lambda: AggressiveBot(),
        "Prob": lambda: ProbabilisticBot(),
        "Conservative": lambda: ConservativeBot(),
    }
    wins, played = tournament(factories, games_per_match=30, players_per_game=4)
    print_results(wins, played)
