"""Final validation of tuned BluffBot config."""
import random
import time
from multiprocessing import Pool
from game import Game, PlayerState
from best_bot import build_best_bot, BEST_CONFIG
from bluffbot import BluffBot


TUNED = {"endgame_dice_threshold": 3, "bluff_below_face": 4, "bluff_to_face": 6}


def match(args):
    seed, dice = args
    rng = random.Random(seed)
    b = BluffBot(**TUNED, **BEST_CONFIG)
    m = build_best_bot()
    if rng.random() < 0.5:
        ps = [PlayerState("B", bot=b), PlayerState("M", bot=m)]
        bi = 0
    else:
        ps = [PlayerState("M", bot=m), PlayerState("B", bot=b)]
        bi = 1
    random.seed(seed * 31 + 17)
    g = Game(ps, starting_dice=dice)
    return 1 if g.play() == bi else 0


def main():
    for d in [5, 3, 1]:
        n = 20000 if d == 5 else 8000
        args = [(i, d) for i in range(n)]
        start = time.time()
        with Pool() as pool:
            wins = sum(pool.map(match, args))
        rate = wins / n
        ci = 1.96 * (rate * (1 - rate) / n) ** 0.5
        verdict = "✓ wins" if rate - ci > 0.5 else ("≈ tie" if rate + ci > 0.5 else "✗ loses")
        print(f"  {d}-dice start: Bluff {wins}/{n} = {rate*100:.2f}% ± {ci*100:.2f}% "
              f" in {time.time()-start:.1f}s  {verdict}")


if __name__ == "__main__":
    print(f"Validating BluffBot {TUNED} vs MasterBot (variant rules, 1v1):")
    main()
