"""BluffBot: MasterBot + endgame ceiling-bluff heuristic.

Idea: when opening a round in a late-game state (both players ≤2 dice), if my
strongest face is low (e.g. I'd be bidding "1×2"), opt instead for a ceiling
bluff — bid 1×6 (or near-ceiling). This forces the opponent into either:
  (a) calling LIAR (which is correct most of the time if my hand is bad), or
  (b) a 2×anything raise that's usually false (since their 1 die can contribute
      at most 1, and they'd need 2 dice to count).

The bluff is a coin-flip: works when opponent's hand has the bluffed face/wild
or when opponent picks (b) instead of (a). It mixes the strategy enough to be
unexploitable on the truthful side.
"""
from iterate2 import MasterBot
from game import Bid, GameState
from bots import count_face_in_hand


class BluffBot(MasterBot):
    name = "Bluff"

    def __init__(self,
                 endgame_dice_threshold: int = 2,   # "endgame" when max alive count ≤ this
                 bluff_below_face: int = 4,         # bluff if my best face is < this
                 bluff_to_face: int = 6,            # what face to bluff to
                 **kwargs):
        super().__init__(**kwargs)
        self.endgame_dice_threshold = endgame_dice_threshold
        self.bluff_below_face = bluff_below_face
        self.bluff_to_face = bluff_to_face

    def _should_endgame_bluff(self, state: GameState) -> bool:
        """Bluff if I'm opening a late-game round with weak dice."""
        if state.current_bid is not None:
            return False
        alive_counts = [c for c in state.dice_counts if c > 0]
        if len(alive_counts) != 2:
            return False
        if max(alive_counts) > self.endgame_dice_threshold:
            return False
        # My best face (highest face I hold, treating 1s as wild)
        best_face = 0
        for f in range(6, 1, -1):
            if count_face_in_hand(state.my_dice, f) > 0:
                best_face = f
                break
        # If my dice are all 1s (wilds), I'm strong — don't bluff
        if best_face == 0 and any(d == 1 for d in state.my_dice):
            return False
        return best_face < self.bluff_below_face

    def act(self, state: GameState):
        if self._should_endgame_bluff(state):
            return Bid(1, self.bluff_to_face)
        return super().act(state)
