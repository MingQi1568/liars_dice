"""Bot strategies for Liar's Dice (1s wild)."""
import random
from math import comb
from game import Bot, Bid, GameState


# ---------- Probability helpers ----------
def binom_at_least(n: int, k: int, p: float) -> float:
    """P(X >= k) where X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # 1 - P(X <= k-1)
    total = 0.0
    for i in range(k):
        total += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return max(0.0, 1.0 - total)


def count_face_in_hand(dice: list[int], face: int) -> int:
    """Count `face` in hand. Includes 1s as wild unless face is 1."""
    if face == 1:
        return sum(1 for d in dice if d == 1)
    return sum(1 for d in dice if d == face or d == 1)


def prob_bid_valid(state: GameState, bid: Bid) -> float:
    """Prob that there are >= bid.quantity dice of bid.face across all dice (1s wild)."""
    in_hand = count_face_in_hand(state.my_dice, bid.face)
    hidden = state.total_dice - len(state.my_dice)
    need = bid.quantity - in_hand
    p_each = 1/6 if bid.face == 1 else 2/6
    return binom_at_least(hidden, need, p_each)


def next_bid_options(current: Bid | None) -> list[Bid]:
    """All bids strictly greater than `current` under the house rule.

    1s are wild for bids on faces 2-6, but a bid on 1s only counts actual 1s.
    Switching between 1s and non-1 (in either direction) requires +2 quantity.
    Same kind: qty strictly higher (or for non-1, same qty with higher face).
    """
    options = []
    if current is None:
        # Opening bid: any face 1-6, any quantity
        for q in range(1, 30):
            for f in range(2, 7):
                options.append(Bid(q, f))
            options.append(Bid(q, 1))
        return options

    if current.face == 1:
        # Same kind (1s): higher qty
        for q in range(current.quantity + 1, current.quantity + 30):
            options.append(Bid(q, 1))
        # Switch to non-1: same qty allowed, any face 2-6
        for q in range(current.quantity, current.quantity + 30):
            for f in range(2, 7):
                options.append(Bid(q, f))
    else:
        # Same quantity, higher face (still non-1)
        for f in range(current.face + 1, 7):
            options.append(Bid(current.quantity, f))
        # Higher quantity, any non-1 face
        for q in range(current.quantity + 1, current.quantity + 30):
            for f in range(2, 7):
                options.append(Bid(q, f))
        # Switch to 1s: need qty strictly higher
        min_1s_qty = current.quantity + 1
        for q in range(min_1s_qty, min_1s_qty + 30):
            options.append(Bid(q, 1))
    return options


# ---------- Bots ----------
class RandomBot(Bot):
    name = "Random"

    def act(self, state: GameState):
        if state.current_bid is None:
            # Must bid
            return Bid(random.randint(1, 3), random.randint(2, 6))
        # 30% chance to call liar
        if random.random() < 0.3:
            return "liar"
        options = next_bid_options(state.current_bid)
        # Restrict to "reasonable" - not way over total dice
        options = [b for b in options if b.quantity <= state.total_dice]
        if not options:
            return "liar"
        return random.choice(options)


class NaiveBot(Bot):
    """Only bids what it actually sees. Calls liar as soon as bid exceeds its hand count + average expectation."""
    name = "Naive"

    def act(self, state: GameState):
        # Find most plentiful face in own hand
        counts = {f: count_face_in_hand(state.my_dice, f) for f in range(2, 7)}
        best_face = max(counts, key=lambda f: counts[f])
        my_count = counts[best_face]

        if state.current_bid is None:
            return Bid(max(1, my_count), best_face)

        # Call liar if current bid quantity > my count + (hidden / 3)
        hidden = state.total_dice - len(state.my_dice)
        expected_total = count_face_in_hand(state.my_dice, state.current_bid.face) + hidden / 3
        if state.current_bid.quantity > expected_total:
            return "liar"

        # Otherwise raise minimally on best face
        cb = state.current_bid
        if best_face > cb.face:
            return Bid(cb.quantity, best_face)
        else:
            return Bid(cb.quantity + 1, best_face)


class ProbabilisticBot(Bot):
    """Calls liar when current bid probability < threshold; bids the highest-EV minimal raise."""
    name = "Prob"

    def __init__(self, call_threshold: float = 0.3, bid_threshold: float = 0.5):
        self.call_threshold = call_threshold
        self.bid_threshold = bid_threshold

    def act(self, state: GameState):
        if state.current_bid is not None:
            p = prob_bid_valid(state, state.current_bid)
            if p < self.call_threshold:
                return "liar"

        # Pick the next bid with the highest probability of being valid, above bid_threshold
        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice]
        # Score: probability bid is valid, prefer smaller quantity bumps
        scored = [(prob_bid_valid(state, b), -b.quantity, -b.face, b) for b in options]
        # Find bids that are safe-ish
        safe = [s for s in scored if s[0] >= self.bid_threshold]
        if safe:
            safe.sort(reverse=True)
            return safe[0][3]
        # Nothing safe - if current bid is risky enough, call liar
        if state.current_bid is not None:
            return "liar"
        # Must bid - pick best probability
        scored.sort(reverse=True)
        return scored[0][3]


class AggressiveBot(Bot):
    """Bluffs by raising aggressively. Threshold for calling liar is low."""
    name = "Aggressive"

    def act(self, state: GameState):
        if state.current_bid is not None:
            p = prob_bid_valid(state, state.current_bid)
            if p < 0.15:
                return "liar"

        # Pick a high-face raise to lock face
        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice]
        # Prefer raising the face on the same quantity if it's a face we have
        if state.current_bid is not None:
            for f in range(state.current_bid.face + 1, 7):
                if count_face_in_hand(state.my_dice, f) >= 1:
                    return Bid(state.current_bid.quantity, f)
        # Otherwise pick a higher-quantity bid on our best face
        counts = {f: count_face_in_hand(state.my_dice, f) for f in range(2, 7)}
        best_face = max(counts, key=lambda f: counts[f])
        q = (state.current_bid.quantity + 1) if state.current_bid else max(1, counts[best_face] + 1)
        return Bid(q, best_face)


class ConservativeBot(Bot):
    """Calls liar quickly. Makes minimal bids."""
    name = "Conservative"

    def act(self, state: GameState):
        if state.current_bid is not None:
            p = prob_bid_valid(state, state.current_bid)
            if p < 0.55:
                return "liar"
        # Make minimal raise on best face
        counts = {f: count_face_in_hand(state.my_dice, f) for f in range(2, 7)}
        best_face = max(counts, key=lambda f: counts[f])
        if state.current_bid is None:
            return Bid(max(1, counts[best_face]), best_face)
        cb = state.current_bid
        if best_face > cb.face and counts[best_face] >= 1:
            return Bid(cb.quantity, best_face)
        return Bid(cb.quantity + 1, best_face)


class SmartBot(Bot):
    """Combines probability with awareness of position/players left.

    - Calls liar based on adaptive threshold considering bid quantity vs total dice.
    - Prefers raising face over quantity when possible.
    - Bluffs more in 1v1 endgame.
    """
    name = "Smart"

    def act(self, state: GameState):
        my_dice = state.my_dice
        hidden = state.total_dice - len(my_dice)

        # --- Liar call logic ---
        if state.current_bid is not None:
            cb = state.current_bid
            p_valid = prob_bid_valid(state, cb)
            # Adaptive threshold: more aggressive in calling when current bid is high relative to total dice
            ratio = cb.quantity / max(1, state.total_dice)
            # Default threshold ~0.35; higher ratio => more willing to call
            threshold = 0.35
            if ratio > 0.6:
                threshold = 0.45
            if p_valid < threshold:
                return "liar"

        # --- Bid logic ---
        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice + 1]

        # Compute probability of validity for each option
        scored = []
        for b in options:
            p = prob_bid_valid(state, b)
            in_hand = count_face_in_hand(my_dice, b.face)
            # Prefer bids that lean on dice we actually have
            hand_bonus = 0.05 * in_hand
            # Prefer minimal escalation (smaller quantity bumps)
            qty_penalty = 0.02 * (b.quantity - (state.current_bid.quantity if state.current_bid else 0))
            score = p + hand_bonus - qty_penalty
            scored.append((score, p, b))

        # Filter to bids that have decent probability of being true
        decent = [(s, p, b) for (s, p, b) in scored if p >= 0.55]
        if decent:
            decent.sort(reverse=True)
            return decent[0][2]

        # If no decent bid, must bluff or call liar
        if state.current_bid is not None:
            # Decide: how risky is the current bid? If it's plausible, we must bluff.
            p_current = prob_bid_valid(state, state.current_bid)
            if p_current >= 0.5:
                # Current bid is plausible; we need to bluff. Pick highest-prob option.
                scored.sort(reverse=True)
                return scored[0][2]
            # Borderline - call liar
            return "liar"

        # Opening bid - safe one
        scored.sort(reverse=True)
        return scored[0][2]
