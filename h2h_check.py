"""Head-to-head: new best_bot config vs old MasterBot defaults."""
import random
import time
from multiprocessing import Pool
from game import Game, PlayerState
from iterate2 import MasterBot
from best_bot import BEST_CONFIG


OLD_CONFIG = {
    "call_threshold": 0.30, "bid_threshold": 0.50, "signal_boost": 0.10,
    "high_ratio_bonus": 0.10, "ratio_breakpoint": 0.6,
    "low_dice_call_bonus": -0.08, "is_1v1_bonus": -0.05,
}


def play_one(seed):
    rng = random.Random(seed)
    new_bot = MasterBot(**BEST_CONFIG)
    old_bot = MasterBot(**OLD_CONFIG)
    if rng.random() < 0.5:
        players = [PlayerState("NEW", bot=new_bot), PlayerState("OLD", bot=old_bot)]
        new_idx = 0
    else:
        players = [PlayerState("OLD", bot=old_bot), PlayerState("NEW", bot=new_bot)]
        new_idx = 1
    random.seed(seed * 31 + 17)
    g = Game(players, starting_dice=5)
    w = g.play()
    return 1 if w == new_idx else 0


def main():
    N = 4000
    start = time.time()
    with Pool() as pool:
        wins = sum(pool.map(play_one, range(N)))
    elapsed = time.time() - start
    print(f"New vs Old MasterBot (1v1, {N} games): {wins}-{N-wins} ({wins/N*100:.2f}%)")
    print(f"95% CI: +/- {1.96*(0.5*0.5/N)**0.5*100:.2f}%   in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
