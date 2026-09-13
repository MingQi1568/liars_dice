"""Second iteration: validate signal bot dominance and tune further."""
import random
from game import Bot, Bid, GameState
from bots import (
    RandomBot, NaiveBot, ProbabilisticBot, AggressiveBot, ConservativeBot, SmartBot,
    next_bid_options, count_face_in_hand, prob_bid_valid, binom_at_least,
)
from iterate import SmartBotV2, SignalBot, eval_variant
from simulate import tournament, print_results, head_to_head


class MasterBot(Bot):
    """Combines Signal's adjusted probability with extra refinements:

    - Signal-adjusted probability for calling LIAR (opponent bids leak info on holdings).
    - Prefers face raises over quantity raises (cheaper escalation).
    - More conservative when low on dice (1-2 left).
    - More aggressive in 1v1 endgame.
    - Slight penalty against bidding on faces it has zero of (avoids signaling weakness).
    """
    name = "Master"

    def __init__(self,
                 call_threshold: float = 0.30,
                 bid_threshold: float = 0.55,
                 signal_boost: float = 0.08,
                 high_ratio_bonus: float = 0.10,
                 ratio_breakpoint: float = 0.6,
                 low_dice_call_bonus: float = 0.08,
                 is_1v1_bonus: float = 0.05):
        # All bonuses are ADDED to the threshold. Higher threshold = more eager to call LIAR
        # (because the rule is `if p < threshold: call_liar`, so a higher threshold catches
        # more bids in the call zone). Pass negative values to make the bot LESS eager in
        # that situation. Grid search determines the best signs.
        self.call_threshold = call_threshold
        self.bid_threshold = bid_threshold
        self.signal_boost = signal_boost
        self.high_ratio_bonus = high_ratio_bonus
        self.ratio_breakpoint = ratio_breakpoint
        self.low_dice_call_bonus = low_dice_call_bonus
        self.is_1v1_bonus = is_1v1_bonus

    def signal_adjusted_prob(self, state: GameState, bid: Bid) -> float:
        in_hand = count_face_in_hand(state.my_dice, bid.face)
        hidden = state.total_dice - len(state.my_dice)
        need = bid.quantity - in_hand
        p_each = 1/6 if bid.face == 1 else 2/6
        signal_count = sum(1 for bidder, b in state.bid_history if bidder != state.my_index and b.face == bid.face)
        adj = min(0.15, self.signal_boost * signal_count)
        p_adj = min(0.99, p_each + adj)
        return binom_at_least(hidden, need, p_adj)

    def act(self, state: GameState):
        my_dice = state.my_dice
        my_count_total = len(my_dice)
        alive = sum(1 for c in state.dice_counts if c > 0)
        is_1v1 = (alive == 2)

        # Calculate dynamic call threshold (higher threshold = more eager to call LIAR)
        threshold = self.call_threshold
        if state.current_bid is not None:
            ratio = state.current_bid.quantity / max(1, state.total_dice)
            if ratio > self.ratio_breakpoint:
                threshold += self.high_ratio_bonus
        if my_count_total <= 2:
            threshold += self.low_dice_call_bonus
        if is_1v1:
            threshold += self.is_1v1_bonus

        # --- Call LIAR check ---
        if state.current_bid is not None:
            p = self.signal_adjusted_prob(state, state.current_bid)
            if p < threshold:
                return "liar"

        # --- Pick best bid ---
        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice + 1]

        scored = []
        for b in options:
            p = self.signal_adjusted_prob(state, b)
            in_hand = count_face_in_hand(my_dice, b.face)
            # Bonus for using faces we actually have (cheaper bluff)
            hand_bonus = 0.05 * in_hand
            # Penalty for raising quantity (vs raising face on same quantity)
            qty_jump = b.quantity - (state.current_bid.quantity if state.current_bid else 0)
            qty_penalty = 0.03 * qty_jump
            # Slight extra penalty for bidding faces we have NONE of (telegraphs bluff if next player notices)
            zero_penalty = 0.04 if in_hand == 0 else 0
            score = p + hand_bonus - qty_penalty - zero_penalty
            scored.append((score, p, b))

        bid_thresh = self.bid_threshold
        # When low on dice, demand higher safety in bids
        if my_count_total <= 2:
            bid_thresh += 0.05

        decent = [(s, p, b) for (s, p, b) in scored if p >= bid_thresh]
        if decent:
            decent.sort(reverse=True)
            return decent[0][2]

        # No "decent" bid available
        if state.current_bid is not None:
            p_current = self.signal_adjusted_prob(state, state.current_bid)
            # If current bid is plausible, we have to bluff
            if p_current >= 0.5:
                scored.sort(reverse=True)
                return scored[0][2]
            return "liar"

        scored.sort(reverse=True)
        return scored[0][2]


def main():
    random.seed(11)

    print("=" * 60)
    print("Fine-tuning SignalBot (250 games each, 4-player mixed opponents)")
    print("=" * 60)
    opponents = [
        lambda: SmartBot(),
        lambda: AggressiveBot(),
        lambda: ProbabilisticBot(),
        lambda: ConservativeBot(),
    ]
    print(f"{'ct':>5} {'bt':>5} {'sb':>5} {'win%':>7}")
    sig_results = []
    for ct in [0.28, 0.33, 0.38]:
        for bt in [0.50, 0.55, 0.60]:
            for sb in [0.06, 0.08, 0.10, 0.12]:
                wr = eval_variant(
                    lambda ct=ct, bt=bt, sb=sb: SignalBot(call_threshold=ct, bid_threshold=bt, signal_boost=sb),
                    opponents, games=250,
                )
                sig_results.append((wr, ct, bt, sb))
                print(f"{ct:>5} {bt:>5} {sb:>5} {wr*100:>6.2f}%", flush=True)
    sig_results.sort(reverse=True)
    print("\nTop 5 Signal configs:")
    for wr, ct, bt, sb in sig_results[:5]:
        print(f"  ct={ct}, bt={bt}, sb={sb} -> {wr*100:.2f}%")

    print("\n" + "=" * 60)
    print("Tuning MasterBot (250 games each)")
    print("=" * 60)
    print(f"{'ct':>5} {'bt':>5} {'sb':>5} {'win%':>7}")
    master_results = []
    for ct in [0.25, 0.30, 0.35]:
        for bt in [0.50, 0.55, 0.60]:
            for sb in [0.06, 0.08, 0.10]:
                wr = eval_variant(
                    lambda ct=ct, bt=bt, sb=sb: MasterBot(call_threshold=ct, bid_threshold=bt, signal_boost=sb),
                    opponents, games=250,
                )
                master_results.append((wr, ct, bt, sb))
                print(f"{ct:>5} {bt:>5} {sb:>5} {wr*100:>6.2f}%", flush=True)
    master_results.sort(reverse=True)
    print("\nTop 5 Master configs:")
    for wr, ct, bt, sb in master_results[:5]:
        print(f"  ct={ct}, bt={bt}, sb={sb} -> {wr*100:.2f}%")

    # Final tournament with fixed naming
    print("\n" + "=" * 60)
    print("Final tournament (4-player, 60 games per combo)")
    print("=" * 60)
    best_sig = sig_results[0]
    best_master = master_results[0]
    factories = {
        "Master_tuned": lambda bs=best_master: MasterBot(call_threshold=bs[1], bid_threshold=bs[2], signal_boost=bs[3]),
        "Signal_tuned": lambda bs=best_sig: SignalBot(call_threshold=bs[1], bid_threshold=bs[2], signal_boost=bs[3]),
        "Smart": lambda: SmartBot(),
        "Aggressive": lambda: AggressiveBot(),
        "Prob": lambda: ProbabilisticBot(),
        "Conservative": lambda: ConservativeBot(),
    }
    wins, played = tournament(factories, games_per_match=60, players_per_game=4)
    print_results(wins, played)

    print("\n=== Head-to-head between top variants (500 games each) ===")
    pairs = [
        ("Master_tuned", "Signal_tuned"),
        ("Master_tuned", "Smart"),
        ("Master_tuned", "Aggressive"),
        ("Signal_tuned", "Aggressive"),
    ]
    for a, b in pairs:
        result = head_to_head(factories[a], factories[b], label_a=a, label_b=b, games=500)
        print(f"  {a} vs {b}: {dict(result)}")


if __name__ == "__main__":
    main()
