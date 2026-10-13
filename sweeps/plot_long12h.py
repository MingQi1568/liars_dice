"""Plots for sweeps/long12h.sh: NashConv over training for the batch-size x Δm runs (by training step and by
games played) and for the CFR+-initialized run.

    python3 sweeps/plot_long12h.py      # writes checkpoints/long12h/nashconv_batch.png and nashconv_cfr_init.png
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"eval @ step (\d+) .*NashConv=([0-9.]+) NashConv_finetuned=([0-9.]+)")
L, B = "checkpoints/long12h/logs", "checkpoints/b4096_long/logs"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def load(*paths):
    out = {}
    for p in paths:                                    # later files (continuations) extend earlier ones
        out.update({int(m[1]): (float(m[2]), float(m[3])) for m in PAT.finditer(open(p).read())})
    return sorted((s, a, b) for s, (a, b) in out.items())


def style(ax, title, xlabel, ymax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0.043, color=INK2, lw=1, ls="--")
    ax.text(0.01, 0.043, " network floor 0.043", color=INK2, fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, ymax)


def label_ends(ax, ends, gap):
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for x, y, lab in ends:
        ax.annotate(lab, xy=(x, y), xytext=(6, 0), textcoords="offset points", fontsize=8.3, color=INK, va="center")


def plot_batch():
    # color follows the entity: batch 4096 / Δm 50k (the control) is blue in every plot of this series
    runs = {
        "batch 4096, Δm 50k": ("#2a78d6", 4096, load(f"{B}/dm50k_s1.log", f"{L}/b4096_dm50k.log")),
        "batch 4096, Δm 10k": ("#eb6834", 4096, load(f"{B}/dm10k_s1.log")),
        "batch 16384, Δm 50k": ("#1baf7a", 16384, load(f"{L}/b16384_dm50k.log")),
        "batch 16384, Δm 10k": ("#eda100", 16384, load(f"{L}/b16384_dm10k.log")),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), facecolor=SURF)
    for row, (xname, xf) in enumerate((("training step (thousands)", lambda s, g: s / 1e3),
                                      ("games played (millions)", lambda s, g: s * g / 1e6))):
        for col, (k, title) in enumerate(((1, "Raw NashConv"), (2, "Fine-tuned NashConv"))):
            ax = axes[row, col]
            style(ax, f"{title} by {xname.split(' (')[0]}", xname, 0.8)
            ends = []
            for name, (c, g, d) in runs.items():
                if not d:
                    continue
                xs, ys = [xf(s, g) for s, *_ in d], [x[k] for x in d]
                ax.plot(xs, ys, color=c, lw=2, solid_capstyle="round")
                ends.append([xs[-1], ys[-1], name])
            label_ends(ax, ends, 0.03)
            ax.set_xlim(0, max(e[0] for e in ends) * 1.32)
    axes[0, 0].set_ylabel("NashConv (lower is better)")
    axes[1, 0].set_ylabel("NashConv (lower is better)")
    handles = [plt.Line2D([], [], color=r[0], lw=2) for r in runs.values()]
    fig.legend(handles, list(runs), loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("1-die R-NaD, seed 1: batch size × reset interval (Δm)", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig("checkpoints/long12h/nashconv_batch.png", dpi=150, facecolor=SURF)


def plot_cfr_init():
    runs = {
        "batch 4096, Δm 50k, from scratch": ("#2a78d6", load(f"{B}/dm50k_s1.log", f"{L}/b4096_dm50k.log")),
        "batch 4096, Δm 50k, from CFR+": ("#eb6834", load(f"{L}/cfr_init.log")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF)
    for ax, k, title in ((axes[0], 1, "Raw NashConv"), (axes[1], 2, "Fine-tuned NashConv")):
        style(ax, title, "training step (thousands)", 0.8)
        ends = []
        for name, (c, d) in runs.items():
            xs, ys = [0.0] + [s / 1e3 for s, *_ in d], None
            if name.endswith("CFR+"):                  # step 0 of the CFR+ start, measured on the saved checkpoint
                ys = [(0.0376, 0.0543)[k - 1]] + [x[k] for x in d]
            else:
                xs, ys = xs[1:], [x[k] for x in d]
            ax.plot(xs, ys, color=c, lw=2, solid_capstyle="round")
            ends.append([xs[-1], ys[-1], name.split(", ")[-1]])
        label_ends(ax, ends, 0.03)
        ax.set_xlim(0, max(e[0] for e in ends) * 1.25)
    axes[0].set_ylabel("NashConv (lower is better)")
    handles = [plt.Line2D([], [], color=r[0], lw=2) for r in runs.values()]
    fig.legend(handles, list(runs), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("1-die R-NaD started at the CFR+ equilibrium (NashConv 0.038) vs from scratch", x=0.01, ha="left",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig("checkpoints/long12h/nashconv_cfr_init.png", dpi=150, facecolor=SURF)


if __name__ == "__main__":
    plot_batch()
    plot_cfr_init()
    print("wrote checkpoints/long12h/nashconv_batch.png and checkpoints/long12h/nashconv_cfr_init.png")
