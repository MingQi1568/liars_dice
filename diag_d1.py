"""Exact diagnostics for R-NaD on the 1-die game (see RNAD_FINDINGS.md, "Phase 1 diagnostics").

    python3 diag_d1.py cfr       [iters]                    # CFR+ reference equilibrium -> cfr_d1_policy.npy
    python3 diag_d1.py oracle    eta inner outer [visit]    # exact tabular R-NaD, NashConv per outer iteration
    python3 diag_d1.py floor     face|infoset [steps]       # distill the CFR+ policy into a network: its NashConv floor
    python3 diag_d1.py patch     CHECKPOINT                 # where is a trained policy exploitable? (needs cfr)
    python3 diag_d1.py finetune  CHECKPOINT                 # NashConv raw / thresholded / thresholded+discretized
"""
import sys
import time

import numpy as np
import torch

from exact_d1 import Tree, discretize_policy, rnad_exact, threshold_policy
from exploit_d1 import make_state, net_prob_fn
from vec_env import Spec, py_features

SPEC = Spec(dice=1)
TREE = Tree(SPEC.qmax)
CFR_PATH = "cfr_d1_policy.npy"


def states():
    return [make_state(h, d) for h in TREE.hists for d in range(1, 7)]


def ckpt_policy(path):
    from rnad import load_policy_net
    net, cfg = load_policy_net(path)
    pi = net_prob_fn(net, cfg.spec)(states()).reshape(TREE.H, 6, TREE.A).astype(np.float64)
    return pi / pi.sum(-1, keepdims=True)


def cmd_cfr(iters=2000):
    from cfr_d1 import CFR
    c, t0 = CFR(SPEC), time.time()
    for k in range(iters // 100):
        c.iterate(100)
        if (k + 1) % 5 == 0:
            print(f"CFR+ {c.t}: NashConv {TREE.nash_conv(c.average_policy())[0]:.5f} [{time.time() - t0:.0f}s]", flush=True)
    np.save(CFR_PATH, c.average_policy())


def cmd_oracle(eta, inner, outer, weighting="infoset"):
    t0 = time.time()
    rnad_exact(TREE, eta, outer, inner, lr=0.5, weighting=weighting,
               log=lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True))


def cmd_floor(kind, steps=3000):
    from rnad_net import InfoSetNet, RNaDNet
    torch.manual_seed(0)
    target = np.load(CFR_PATH)
    static, win, wmask, mask = (torch.stack(x) for x in zip(*[py_features(s, SPEC) for s in states()]))
    tgt = torch.tensor(target.reshape(-1, TREE.A), dtype=torch.float32)
    net = InfoSetNet(SPEC, 256) if kind == "infoset" else RNaDNet(SPEC)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3 if kind == "infoset" else 2e-3)
    for k in range(1, steps + 1):
        logp = torch.log_softmax(net.policy_logits(static, win, wmask, mask).masked_fill(~mask, float("-inf")), -1)
        kl = (tgt * (torch.log(tgt.clamp(min=1e-12)) - logp)).masked_fill(~mask, 0.0).sum(-1).mean()
        opt.zero_grad()
        kl.backward()
        opt.step()
        if k % (steps // 4) == 0:
            p = logp.detach().exp().numpy().astype(np.float64).reshape(TREE.H, 6, TREE.A)
            print(f"{kind} step {k}: KL {kl.item():.5f} NashConv {TREE.nash_conv(p / p.sum(-1, keepdims=True))[0]:.4f}",
                  flush=True)


def cmd_patch(path):
    star, pi = np.load(CFR_PATH), ckpt_policy(path)
    vis = TREE.visitation(pi)
    print(f"policy NashConv {TREE.nash_conv(pi)[0]:.4f}, CFR+ {TREE.nash_conv(star)[0]:.4f}")
    for label, sel in (("rarely visited (vis < tau)", lambda t: vis < t), ("main line (vis >= tau)", lambda t: vis >= t)):
        print(f"replace with CFR+ at {label} infosets:")
        for tau in (1e-5, 1e-4, 1e-3, 1e-2):
            m = sel(tau)
            nc = TREE.nash_conv(np.where(m[:, :, None], star, pi))[0]
            print(f"  tau {tau:.0e}: {m.mean():6.1%} of infosets, {vis[m].sum() / vis.sum():6.2%} of visits -> NashConv {nc:.4f}")


def cmd_finetune(path):
    pi = ckpt_policy(path)
    thr = threshold_policy(pi, TREE.legal, 0.03)
    print(f"raw {TREE.nash_conv(pi)[0]:.4f} | threshold 0.03 {TREE.nash_conv(thr)[0]:.4f} | "
          f"+ discretize/32 {TREE.nash_conv(discretize_policy(thr, 32))[0]:.4f}")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "cfr":
        cmd_cfr(*(int(x) for x in rest))
    elif cmd == "oracle":
        cmd_oracle(float(rest[0]), int(rest[1]), int(rest[2]), *(rest[3:4]))
    elif cmd == "floor":
        cmd_floor(rest[0], *(int(x) for x in rest[1:2]))
    elif cmd == "patch":
        cmd_patch(rest[0])
    elif cmd == "finetune":
        cmd_finetune(rest[0])
    else:
        print(__doc__)
