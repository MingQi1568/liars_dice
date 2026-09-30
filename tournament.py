"""Round-robin tournament between NFSP checkpoints saved at different training episodes.
Plays --games-per-pair games for every pair (alternating who opens each game to cancel
first-player advantage) and writes a win-rate grid to tournament_results.json.

Usage:
    python3 tournament.py --episodes 20000 40000 ... --games-per-pair 1000
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from game import Game, PlayerState  # noqa: E402
from nfsp_model import NFSPBot  # noqa: E402

_bots = {}


def _init_worker(checkpoint_dir, episodes):
    import torch
    torch.set_num_threads(1)
    for ep in episodes:
        _bots[ep] = NFSPBot.from_checkpoint(os.path.join(checkpoint_dir, f"nfsp_ep{ep}.pt"), device="cpu", sample=True)


def _play_one(args):
    ep_a, ep_b, a_first = args
    bot_a, bot_b = _bots[ep_a], _bots[ep_b]
    if a_first:
        players = [PlayerState("A", bot=bot_a), PlayerState("B", bot=bot_b)]
    else:
        players = [PlayerState("B", bot=bot_b), PlayerState("A", bot=bot_a)]
    g = Game(players, starting_dice=5, verbose=False)
    w = g.play()
    return ep_a, ep_b, players[w].name == "A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--games-per-pair", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/shared_face")
    parser.add_argument("--out", type=str, default="tournament_results.json")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--vs", type=int, default=None,
                         help="if set, only play this episode against every other in --episodes "
                              "(a single-row comparison) instead of a full round robin")
    args = parser.parse_args()

    episodes = args.episodes
    if args.vs is not None:
        others = [e for e in episodes if e != args.vs]
        pairs = [(args.vs, e) for e in others]
        if args.vs not in episodes:
            episodes = episodes + [args.vs]
    else:
        pairs = [(ep_a, ep_b) for i, ep_a in enumerate(episodes) for ep_b in episodes[i + 1:]]

    tasks = []
    for ep_a, ep_b in pairs:
        for k in range(args.games_per_pair):
            tasks.append((ep_a, ep_b, k % 2 == 0))
    print(f"{len(episodes)} checkpoints, {len(pairs)} pairs, {len(tasks)} games total", flush=True)

    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.checkpoint_dir, episodes)) as pool:
        results = pool.map(_play_one, tasks, chunksize=200)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({len(tasks) / elapsed:.0f} games/sec)", flush=True)

    wins = {ep: {ep2: 0 for ep2 in episodes} for ep in episodes}
    games = {ep: {ep2: 0 for ep2 in episodes} for ep in episodes}
    for ep_a, ep_b, a_won in results:
        games[ep_a][ep_b] += 1
        games[ep_b][ep_a] += 1
        if a_won:
            wins[ep_a][ep_b] += 1
        else:
            wins[ep_b][ep_a] += 1

    grid = {
        str(ep): {str(ep2): (wins[ep][ep2] / games[ep][ep2] if games[ep][ep2] else None) for ep2 in episodes}
        for ep in episodes
    }
    payload = {"episodes": episodes, "games_per_pair": args.games_per_pair, "grid": grid}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {args.out}", flush=True)

    if args.vs is not None:
        print(f"\n{args.vs // 1000}k win rate vs each opponent:")
        for e in others:
            print(f"  vs {e // 1000:>4}k: {grid[str(args.vs)][str(e)] * 100:5.1f}%")
    else:
        header = "        " + "".join(f"{ep // 1000:>6}k" for ep in episodes)
        print(header)
        for ep in episodes:
            row = f"{ep // 1000:>6}k  "
            for ep2 in episodes:
                if ep == ep2:
                    row += "     - "
                else:
                    row += f"{grid[str(ep)][str(ep2)] * 100:5.1f}% "
            print(row)


if __name__ == "__main__":
    main()
