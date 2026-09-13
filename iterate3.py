"""Parallel grid search over all MasterBot parameters, including signs of modifiers.

Phase 1: rough grid over all 6 tunable params, 200 games each.
Phase 2: validate top 10 with 1000 games each.
Phase 3: head-to-head vs current MasterBot defaults.
"""
import os
import sys
import time
import random
import itertools
from multiprocessing import Pool
from game import Game, PlayerState
from iterate2 import MasterBot
from bots import SmartBot, AggressiveBot, ProbabilisticBot, ConservativeBot


# Defined at module level so worker processes can find them when using 'spawn'.
OPPONENT_CLASSES = [SmartBot, AggressiveBot, ProbabilisticBot, ConservativeBot]


def play_one_game(args):
    """Play one 1v1 game, return (cfg_idx, 1 if our bot won else 0)."""
    cfg_idx, seed, cfg = args
    rng = random.Random(seed)
    opp_cls = rng.choice(OPPONENT_CLASSES)
    opp = opp_cls()
    our_bot = MasterBot(**cfg)

    # Randomize seating each game
    if rng.random() < 0.5:
        players = [PlayerState("US", bot=our_bot), PlayerState("OP", bot=opp)]
        us_idx = 0
    else:
        players = [PlayerState("OP", bot=opp), PlayerState("US", bot=our_bot)]
        us_idx = 1

    # Force the engine to use our seeded RNG via random.seed
    random.seed(seed * 7919 + 1)
    g = Game(players, starting_dice=5, verbose=False)
    winner_idx = g.play()
    return cfg_idx, (1 if winner_idx == us_idx else 0)


def run_games(configs, games_per_config, base_seed, workers):
    tasks = []
    for cfg_idx, cfg in enumerate(configs):
        for g in range(games_per_config):
            seed = base_seed + cfg_idx * games_per_config + g
            tasks.append((cfg_idx, seed, cfg))

    print(f"  {len(tasks):,} games on {workers} workers...", flush=True)
    start = time.time()

    chunksize = max(1, len(tasks) // (workers * 40))
    with Pool(processes=workers) as pool:
        results = pool.map(play_one_game, tasks, chunksize=chunksize)

    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s ({len(tasks)/elapsed:,.0f} games/sec)", flush=True)

    wins = [0] * len(configs)
    plays = [0] * len(configs)
    for cfg_idx, win in results:
        wins[cfg_idx] += win
        plays[cfg_idx] += 1

    return [(wins[i]/plays[i], i, configs[i]) for i in range(len(configs))]


def fmt_cfg(cfg):
    return (f"ct={cfg['call_threshold']:.2f} bt={cfg['bid_threshold']:.2f} "
            f"sb={cfg['signal_boost']:.2f} hrb={cfg['high_ratio_bonus']:+.2f} "
            f"ldb={cfg['low_dice_call_bonus']:+.2f} 1v1b={cfg['is_1v1_bonus']:+.2f}")


def main():
    workers = os.cpu_count() or 8
    print(f"=== Grid search on {workers} cores ===\n")

    # Phase 1: coarse grid
    grid_axes = {
        "call_threshold":      [0.15, 0.22, 0.30, 0.37, 0.45],
        "bid_threshold":       [0.40, 0.50, 0.55, 0.62],
        "signal_boost":        [0.05, 0.08, 0.11, 0.14],
        "high_ratio_bonus":    [-0.10, 0.05, 0.15],
        "low_dice_call_bonus": [-0.10, 0.05, 0.15],
        "is_1v1_bonus":        [-0.10, 0.05, 0.15],
    }
    keys = list(grid_axes.keys())
    configs = []
    for vals in itertools.product(*[grid_axes[k] for k in keys]):
        configs.append(dict(zip(keys, vals)))

    print(f"Phase 1: coarse grid of {len(configs):,} configs × 200 games")
    results = run_games(configs, games_per_config=200, base_seed=20260519, workers=workers)
    results.sort(reverse=True)

    print("\nTop 15 from phase 1:")
    print(f"  {'win%':>6}  {'config'}")
    for wr, idx, cfg in results[:15]:
        print(f"  {wr*100:5.2f}%  {fmt_cfg(cfg)}")

    # Phase 2: validate top 25 with more games
    print(f"\nPhase 2: validate top 25 with 1000 games each")
    top25 = [r[2] for r in results[:25]]
    final = run_games(top25, games_per_config=1000, base_seed=99887766, workers=workers)
    final.sort(reverse=True)

    print("\nFinal ranking (top 15):")
    print(f"  {'win%':>6}  {'config'}")
    for wr, idx, cfg in final[:15]:
        print(f"  {wr*100:5.2f}%  {fmt_cfg(cfg)}")

    best_wr, _, best_cfg = final[0]
    print(f"\n*** WINNER: {best_wr*100:.2f}% ***")
    print(f"    {fmt_cfg(best_cfg)}")
    print(f"    raw cfg: {best_cfg}")

    # Phase 3: head-to-head vs old default
    print(f"\nPhase 3: new best vs old MasterBot defaults (2000 games)")
    old_cfg = {
        "call_threshold": 0.30, "bid_threshold": 0.50, "signal_boost": 0.10,
        "high_ratio_bonus": 0.10, "ratio_breakpoint": 0.6,
        "low_dice_call_bonus": -0.08,  # old behavior: subtract — encode as negative
        "is_1v1_bonus": -0.05,
    }
    # Add ratio_breakpoint to best_cfg for fair comparison
    best_cfg_full = dict(best_cfg)
    best_cfg_full["ratio_breakpoint"] = 0.6

    h2h_results = run_games([best_cfg_full, old_cfg], games_per_config=2000,
                            base_seed=55443322, workers=workers)
    new_wr = h2h_results[0][0]
    old_wr = h2h_results[1][0]
    print(f"  New best vs mixed opponents: {new_wr*100:.2f}%")
    print(f"  Old MasterBot vs mixed opponents: {old_wr*100:.2f}%")
    print(f"  Improvement: {(new_wr - old_wr)*100:+.2f} percentage points")


if __name__ == "__main__":
    main()
