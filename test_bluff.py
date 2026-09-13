"""Test BluffBot vs MasterBot in 1v1 (variant rules, first to 0 wins).

Three scenarios:
  1. Full-game: 5 dice each (endgame happens occasionally)
  2. Mid-game:  3 dice each (endgame happens often)
  3. Endgame:   1 die each (every game IS the endgame — direct test of the heuristic)
"""
import random
import time
from multiprocessing import Pool
from game import Game, PlayerState
from best_bot import build_best_bot, BEST_CONFIG
from bluffbot import BluffBot


def play_match(args):
    """One game: BluffBot vs MasterBot. Returns 1 if Bluff won."""
    seed, starting_dice = args
    rng = random.Random(seed)
    bluff = BluffBot(**BEST_CONFIG)
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


def run(starting_dice, n_games=2000, workers=None):
    args = [(i, starting_dice) for i in range(n_games)]
    start = time.time()
    with Pool(processes=workers) as pool:
        results = pool.map(play_match, args)
    elapsed = time.time() - start
    wins = sum(results)
    rate = wins / n_games
    ci = 1.96 * (rate * (1 - rate) / n_games) ** 0.5
    return wins, n_games, rate, ci, elapsed


def main():
    for dice in [5, 3, 1]:
        n = 4000 if dice == 1 else 2000
        wins, total, rate, ci, t = run(dice, n_games=n)
        verdict = ("✓ Bluff wins" if rate - ci > 0.5
                   else "✗ Master wins" if rate + ci < 0.5
                   else "≈ tie (CI overlaps 50%)")
        print(f"  {dice}-dice start: Bluff {wins}/{total} = {rate*100:.2f}%  "
              f"±{ci*100:.2f}%   in {t:.1f}s   {verdict}")


if __name__ == "__main__":
    print("BluffBot vs MasterBot (1v1, variant rules)")
    print("-" * 60)
    main()
