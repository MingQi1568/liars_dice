"""Builder for the best bot.

After three rounds of grid search:

  Iter 1 (iterate.py):
    Tested SmartBotV2 and SignalBot — SignalBot dominated at ~68%.

  Iter 2 (iterate2.py):
    Found MasterBot config (ct=0.30, bt=0.50, sb=0.10) at 69.6% vs mixed opponents.
    Had a bug: low_dice_call_bonus and is_1v1_bonus had sign-vs-intent inconsistency.

  Iter 3 (iterate3.py) — current best:
    Fixed semantics (all bonuses now additive: higher threshold = more eager to call).
    Added is_1v1_bonus as a tunable parameter.
    Parallel grid search over 2,160 configs × 200 games, validated top 25 with 1000 each.
    Best config below at 82.40% vs mixed opponents (1v1, variant rules).
    +2.25 pp over previous defaults.

Key insight from iter 3: the buggy signs in iter 2 turned out to be closer to optimal
than the "fixed" version. The bot performs best when it is LESS eager to call LIAR
when its own dice count is low, and LESS eager in 1v1 — likely because the probability
estimate is noisier with fewer hidden dice to model.
"""
from iterate2 import MasterBot
from bluffbot import BluffBot


BEST_CONFIG = {
    "call_threshold": 0.22,
    "bid_threshold": 0.40,
    "signal_boost": 0.08,
    "high_ratio_bonus": 0.05,
    "ratio_breakpoint": 0.6,
    "low_dice_call_bonus": -0.10,
    "is_1v1_bonus": -0.10,
}

BLUFF_CONFIG = {
    # Trigger ceiling bluff when both players have <= this many dice AND
    # my best non-1 face is below bluff_below_face.
    "endgame_dice_threshold": 3,
    "bluff_below_face": 4,        # bluff if best face is 2 or 3
    "bluff_to_face": 6,           # always bluff to the ceiling
}


def build_best_bot():
    """The best bot: MasterBot + endgame ceiling-bluff heuristic.

    Validated: BluffBot beats vanilla MasterBot ~52.5% at 5-dice 1v1 (20k games),
    ~54% at 1-die endgame (8k games). Both statistically significant.
    """
    return BluffBot(**BLUFF_CONFIG, **BEST_CONFIG)


def build_master_only():
    """Vanilla MasterBot, no bluff heuristic. For comparison / testing."""
    return MasterBot(**BEST_CONFIG)
