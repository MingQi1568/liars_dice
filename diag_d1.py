"""Exact diagnostics for R-NaD on the 1-die game (see RNAD_FINDINGS.md, "Phase 1 diagnostics").

    python3 diag_d1.py cfr       [iters]                    # CFR+ reference equilibrium -> cfr_d1_policy.npy
    python3 diag_d1.py oracle    eta inner outer [visit]    # exact tabular R-NaD, NashConv per outer iteration
    python3 diag_d1.py floor     VARIANT [steps]            # distill the CFR+ policy into a network: its NashConv floor
                                                            # VARIANT: infoset | face | face+rank | face+ctx | face+rank+ctx
    python3 diag_d1.py patch     CHECKPOINT                 # where is a trained policy exploitable? (needs cfr)
    python3 diag_d1.py finetune  CHECKPOINT                 # NashConv raw / thresholded / thresholded+discretized
    python3 diag_d1.py liar      CHECKPOINT|POLICY.npy ...  # is the error in the liar decision or among raises? (needs cfr)
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


def cmd_floor(kind, steps=3000, save=None):
    from rnad_net import InfoSetNet, RNaDNet
    torch.manual_seed(0)
    target = np.load(CFR_PATH)
    static, win, wmask, mask = (torch.stack(x) for x in zip(*[py_features(s, SPEC) for s in states()]))
    tgt = torch.tensor(target.reshape(-1, TREE.A), dtype=torch.float32)
    net = InfoSetNet(SPEC, 256) if kind == "infoset" else \
        RNaDNet(SPEC, face_rank="rank" in kind, info_ctx=64 if "ctx" in kind else 0)
    print(f"{kind}: {sum(p.numel() for p in net.parameters())} parameters", flush=True)
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
    if save:
        np.save(save, p / p.sum(-1, keepdims=True))


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


def load_policy(path):
    return np.load(path) if path.endswith(".npy") else ckpt_policy(path)


def swap_liar(pi, star):
    """pi with its probability of calling liar replaced by star's; raises keep pi's relative preferences."""
    R = TREE.R
    out = pi.copy()
    has = TREE.legal[:, R][:, None] & (pi[..., :R].sum(-1) > 0)
    scale = (1.0 - star[..., R]) / np.maximum(pi[..., :R].sum(-1), 1e-300)
    out[..., :R] = np.where(has[..., None], pi[..., :R] * scale[..., None], pi[..., :R])
    out[..., R] = np.where(has, star[..., R], pi[..., R])
    return out


def swap_raises(pi, star):
    """pi with its choice AMONG raises replaced by star's relative preferences; P(liar) kept from pi."""
    R = TREE.R
    out = pi.copy()
    sr = star[..., :R].sum(-1)
    ok = sr > 1e-12
    mass = 1.0 - pi[..., R]
    out[..., :R] = np.where(ok[..., None], star[..., :R] / np.maximum(sr, 1e-300)[..., None] * mass[..., None], pi[..., :R])
    return out


def cmd_liar(*paths):
    star = np.load(CFR_PATH)
    R = TREE.R
    for path in paths:
        pi = load_policy(path)
        vis = TREE.visitation(pi)
        m = TREE.legal[:, R][:, None] & np.ones((1, 6), bool)
        err = np.abs(pi[..., R] - star[..., R])
        print(f"{path}\n  NashConv {TREE.nash_conv(pi)[0]:.4f} | fix only P(liar) -> {TREE.nash_conv(swap_liar(pi, star))[0]:.4f} | "
              f"fix only the choice among raises -> {TREE.nash_conv(swap_raises(pi, star))[0]:.4f}\n"
              f"  visit-weighted mean |P(liar) - CFR+| = {(vis * err * m).sum() / (vis * m).sum():.4f}", flush=True)


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
        cmd_floor(rest[0], *(int(x) for x in rest[1:2]), *(rest[2:3]))
    elif cmd == "patch":
        cmd_patch(rest[0])
    elif cmd == "liar":
        cmd_liar(*rest)
    elif cmd == "finetune":
        cmd_finetune(rest[0])
    else:
        print(__doc__)
