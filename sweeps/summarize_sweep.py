"""Summarize a 1-die sweep: mean exact NashConv per arm (over seeds) at matched steps, raw [fine-tuned],
plus each arm's mean best value and whether it degraded (final more than 0.05 above its best).

    python3 sweeps/summarize_sweep.py checkpoints/sweep_d1/logs
"""
import collections
import os
import re
import sys

PAT = re.compile(r"eval @ step (\d+) .*NashConv=([0-9.]+) NashConv_finetuned=([0-9.]+)")
CHECK = (50_000, 100_000, 200_000, 350_000, 500_000, 750_000, 1_000_000)


def main(logdir):
    runs = collections.defaultdict(dict)                 # arm -> seed -> {step: (raw, ft)}
    for f in sorted(os.listdir(logdir)):
        if f.endswith(".log"):
            arm, seed = f[:-4].rsplit("_s", 1)
            runs[arm][seed] = {int(m[1]): (float(m[2]), float(m[3])) for m in PAT.finditer(open(os.path.join(logdir, f)).read())}
    head = "| arm | " + " | ".join(f"{s // 1000}k" for s in CHECK) + " | best (mean) | final step | degraded? |"
    print(head + "\n|" + "---|" * (len(CHECK) + 4))
    for arm, seeds in runs.items():
        cells = []
        for s in CHECK:
            vals = [r[s] for r in seeds.values() if s in r]
            cells.append(f"{sum(v[0] for v in vals) / len(vals):.3f} [{sum(v[1] for v in vals) / len(vals):.3f}]"
                         + ("" if len(vals) == len(seeds) else "*") if vals else "—")
        best = [min(v[0] for v in r.values()) for r in seeds.values() if r]
        last = [r[max(r)] for r in seeds.values() if r]
        degraded = any(l[0] - b > 0.05 for l, b in zip(last, best))
        final = min(max(r) for r in seeds.values() if r) if any(seeds.values()) else 0
        print(f"| {arm} | " + " | ".join(cells) + f" | {sum(best) / max(1, len(best)):.3f} | {final // 1000}k | "
              f"{'yes' if degraded else 'no'} |")
    print("\nRaw NashConv [fine-tuned], mean over seeds; * = not all seeds reached that step. "
          "CFR+ 0.0026; DeepMind's code on our game ~0.23 [0.13].")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sweep_d1/logs")
