"""Evaluate NFSP checkpoints against BestBot as they appear, independently of training.

Watches one or more checkpoint directories, plays --games games (seats alternated) for every
nfsp_ep*.pt not yet evaluated, and appends one JSON line per checkpoint to --out. Training is
never touched -- this only reads checkpoint files, which are written once and never modified.
Exits once every --log contains "Training complete." and nothing is left to evaluate.

Usage:
    python3 eval_watch.py --dirs checkpoints/shared_face_liar checkpoints/shared_face_liar_faceid \\
        --logs train_nfsp_shared_face_liar.log train_nfsp_shared_face_liar_faceid.log
"""
import argparse
import glob
import json
import os
import re
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from game import Game, PlayerState  # noqa: E402
from nfsp_model import NFSPBot  # noqa: E402

EP_RE = re.compile(r"nfsp_ep(\d+)\.pt$")
_cache = {}


def _play_chunk(args):
    path, n_games, first_game = args
    import torch
    from best_bot import build_best_bot
    torch.set_num_threads(1)
    if _cache.get("path") != path:
        _cache["path"], _cache["bot"] = path, NFSPBot.from_checkpoint(path, device="cpu", sample=True)
    bot = _cache["bot"]
    wins = 0
    for k in range(first_game, first_game + n_games):
        opp = build_best_bot()
        players = [PlayerState("NFSP", bot=bot), PlayerState("Opp", bot=opp)]
        if k % 2 == 1:
            players.reverse()
        winner = Game(players, starting_dice=bot.cfg.dice_count, verbose=False).play()
        wins += players[winner].name == "NFSP"
    return wins


def pending(dirs, done, min_age):
    now = time.time()
    found = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "nfsp_ep*.pt")):
            m = EP_RE.search(path)
            key = (os.path.basename(os.path.normpath(d)), int(m.group(1)))
            if key not in done and now - os.path.getmtime(path) >= min_age:
                found.append((key[1], key[0], path))
    return sorted(found)


def training_finished(logs):
    for log in logs:
        try:
            with open(log, encoding="utf-8", errors="ignore") as f:
                if "Training complete." not in f.read():
                    return False
        except FileNotFoundError:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--logs", nargs="+", default=[], help="training logs; exit once all say 'Training complete.'")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="eval_bestbot_1000.jsonl")
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--min-age", type=float, default=15.0, help="seconds a checkpoint must be old before reading it")
    args = parser.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["run"], r["episode"]))

    chunk = -(-args.games // args.workers)
    with Pool(processes=args.workers) as pool:
        while True:
            todo = pending(args.dirs, done, args.min_age)
            if not todo:
                if training_finished(args.logs):
                    print("training finished and every checkpoint is evaluated; exiting", flush=True)
                    return
                time.sleep(args.poll)
                continue
            for episode, run, path in todo:
                t0 = time.time()
                tasks = [(path, min(chunk, args.games - s), s) for s in range(0, args.games, chunk)]
                try:
                    wins = sum(pool.map(_play_chunk, tasks))
                except Exception as e:  # e.g. a checkpoint that failed to load; retry next poll
                    print(f"{run} ep {episode}: {type(e).__name__}: {e}; will retry", flush=True)
                    continue
                p = wins / args.games
                rec = {"run": run, "episode": episode, "games": args.games, "wins": wins,
                       "win_rate": p, "se": (p * (1 - p) / args.games) ** 0.5, "time": time.time()}
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done.add((run, episode))
                print(f"{run} ep {episode}: {p * 100:.1f}% vs BestBot ({args.games} games, {time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
