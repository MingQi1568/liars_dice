"""Quick grid over bluff params."""
import random
import time
import itertools
from multiprocessing import Pool
from game import Game, PlayerState
from best_bot import build_best_bot, BEST_CONFIG
from bluffbot import BluffBot


def play_match(args):
    seed, cfg, starting_dice = args
    rng = random.Random(seed)
    bluff = BluffBot(
        endgame_dice_threshold=cfg["endgame"],
        bluff_below_face=cfg["below"],
        bluff_to_face=cfg["to"],
        **BEST_CONFIG,
    )
    master = build_best_bot()
    if rng.random() < 0.5:
        players = [PlayerState("BLUFF", bot=bluff), PlayerState("MASTER", bot=master)]
        bluff_idx = 0
    else:
        players = [PlayerState("MASTER", bot=master), PlayerState("BLUFF", bot=bluff)]
        bluff_idx = 1
    random.seed(seed * 31 + 17)
    g = Game(players, starting_dice=starting_dice)
    w = g.play()
    return 1 if w == bluff_idx else 0


def main():
    grid = list(itertools.product(
        [1, 2, 3],          # endgame_dice_threshold
        [3, 4, 5],          # bluff_below_face
        [4, 5, 6],          # bluff_to_face
    ))
    configs = [{"endgame": e, "below": b, "to": t} for e, b, t in grid]
    n = 1500
    print(f"Tuning over {len(configs)} configs × {n} games at 5-dice start")
    print()
    args = []
    for ci, cfg in enumerate(configs):
        for g in range(n):
            args.append((ci * n + g, cfg, 5))

    start = time.time()
    with Pool() as pool:
        results = pool.map(play_match, args)
    elapsed = time.time() - start

    wins = [0] * len(configs)
    for i, w in enumerate(results):
        wins[i // n] += w

    rows = [(wins[i] / n, i, configs[i]) for i in range(len(configs))]
    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"{'win%':>7}  endgame  below  to-face")
    for rate, _, cfg in rows:
        print(f"  {rate*100:5.2f}%  {cfg['endgame']:7}  {cfg['below']:5}  {cfg['to']:7}")
    print(f"\nFinished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
