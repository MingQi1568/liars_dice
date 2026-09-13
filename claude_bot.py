"""A 'Claude-style' bot — what I'd do if playing thoughtfully.

Key differences vs MasterBot:
- Uses pass-through endorsement: every player who saw a bid of face F and chose
  to keep bidding (rather than calling LIAR) implicitly endorses that face's count.
- Per-opponent tracking: if the same opponent keeps raising face F, they likely
  hold many F-or-wilds.
- Slightly more eager to call LIAR when (current bid quantity)/(total dice) > 0.5.
- Prefers face-raise > quantity-raise (less risk).
- More conservative on bids when down to last 1-2 dice.
"""
from game import Bot, Bid, GameState
from bots import binom_at_least, count_face_in_hand, next_bid_options


class ClaudeBot(Bot):
    name = "Claude"

    def __init__(self,
                 call_threshold: float = 0.30,
                 bid_safety: float = 0.55,
                 raiser_boost: float = 0.10,        # per opponent bid of this face
                 endorsement_boost: float = 0.04,   # per pass-through of this face
                 same_raiser_extra: float = 0.05,   # extra if same player keeps bidding F
                 high_ratio_call_bonus: float = 0.10):
        self.call_threshold = call_threshold
        self.bid_safety = bid_safety
        self.raiser_boost = raiser_boost
        self.endorsement_boost = endorsement_boost
        self.same_raiser_extra = same_raiser_extra
        self.high_ratio_call_bonus = high_ratio_call_bonus

    def adjusted_prob(self, state: GameState, bid: Bid) -> float:
        in_hand = count_face_in_hand(state.my_dice, bid.face)
        hidden = state.total_dice - len(state.my_dice)
        need = bid.quantity - in_hand
        p_each = 1/6 if bid.face == 1 else 2/6

        # Bid-history-based adjustments
        opponents_who_bid_face = set()
        bids_of_face = 0
        same_raiser_count_max = 0
        bids_after_face: dict[int, int] = {}  # per-bidder: how many later bids did they make after seeing this face?

        # Find first appearance of bid.face in history
        first_appearance_idx = None
        for idx, (bidder, b) in enumerate(state.bid_history):
            if bidder == state.my_index:
                continue
            if b.face == bid.face:
                if first_appearance_idx is None:
                    first_appearance_idx = idx
                opponents_who_bid_face.add(bidder)
                bids_of_face += 1

        # Count "endorsements": after the first time face F was bid, any other bid (not on F, not by me) means they passed F
        endorsements = 0
        if first_appearance_idx is not None:
            for idx in range(first_appearance_idx + 1, len(state.bid_history)):
                bidder, b = state.bid_history[idx]
                if bidder == state.my_index:
                    continue
                if b.face != bid.face:
                    endorsements += 1

        # Count repeated raises of face F by same opponent
        raises_by_player: dict[int, int] = {}
        for bidder, b in state.bid_history:
            if bidder == state.my_index:
                continue
            if b.face == bid.face:
                raises_by_player[bidder] = raises_by_player.get(bidder, 0) + 1
        if raises_by_player:
            same_raiser_count_max = max(raises_by_player.values())

        adj = (self.raiser_boost * len(opponents_who_bid_face)
               + self.endorsement_boost * endorsements
               + self.same_raiser_extra * max(0, same_raiser_count_max - 1))
        adj = min(0.20, adj)
        p_adj = min(0.99, p_each + adj)
        return binom_at_least(hidden, need, p_adj)

    def act(self, state: GameState):
        my_count_total = len(state.my_dice)
        alive = sum(1 for c in state.dice_counts if c > 0)
        is_1v1 = (alive == 2)

        # --- Decide whether to call LIAR ---
        if state.current_bid is not None:
            cb = state.current_bid
            ratio = cb.quantity / max(1, state.total_dice)
            threshold = self.call_threshold
            if ratio > 0.5:
                threshold += self.high_ratio_call_bonus
            if my_count_total <= 2:
                threshold -= 0.05
            if is_1v1:
                threshold -= 0.03

            p = self.adjusted_prob(state, cb)
            if p < threshold:
                return "liar"

        # --- Pick a bid ---
        options = next_bid_options(state.current_bid)
        options = [b for b in options if b.quantity <= state.total_dice + 1]

        scored = []
        for b in options:
            p = self.adjusted_prob(state, b)
            in_hand = count_face_in_hand(state.my_dice, b.face)
            # Prefer face-raise on same quantity (cheap)
            same_qty = state.current_bid is not None and b.quantity == state.current_bid.quantity
            same_qty_bonus = 0.06 if same_qty else 0
            hand_bonus = 0.04 * in_hand
            qty_jump = b.quantity - (state.current_bid.quantity if state.current_bid else 0)
            qty_penalty = 0.03 * qty_jump
            zero_penalty = 0.04 if in_hand == 0 else 0
            score = p + hand_bonus + same_qty_bonus - qty_penalty - zero_penalty
            scored.append((score, p, b))

        safety = self.bid_safety
        if my_count_total <= 2:
            safety += 0.05

        decent = [(s, p, b) for (s, p, b) in scored if p >= safety]
        if decent:
            decent.sort(reverse=True)
            return decent[0][2]

        # Must bluff or call
        if state.current_bid is not None:
            p_current = self.adjusted_prob(state, state.current_bid)
            if p_current >= 0.5:
                # Forced to bluff: pick best-scored option
                scored.sort(reverse=True)
                return scored[0][2]
            return "liar"

        # Opening bid (shouldn't reach without current_bid)
        scored.sort(reverse=True)
        return scored[0][2]
