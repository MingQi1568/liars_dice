"""Start R-NaD from the CFR+ equilibrium at 1 die: does self-play keep it there, or drift away?

Builds an R-NaD agent (InfoSetNet), fits its policy to the exact CFR+ policy at every information set (full
batch, as `diag_d1.py floor`), then keeps fitting the policy while also fitting the value head to Monte Carlo
returns of self-play under it (the value R-NaD's v-trace would learn; the regularization penalty is zero
because the policy equals the regularization policy). The fitted network becomes the online net, the target
net and both regularization policies, with a fresh optimizer, saved as a step-0 checkpoint:

    python3 diag_d1.py cfr                                   # once: writes cfr_d1_policy.npy
    python3 init_cfr_d1.py OUT.pt --games-per-step 4096 --sched-sizes 50000
    python3 train_rnad.py --resume OUT.pt --checkpoint-dir ... # then ordinary R-NaD from there
"""
import argparse
import time

import numpy as np
import torch

from diag_d1 import CFR_PATH, SPEC, TREE, states
from rnad import NEG, Config, NetReg, RNaD, collect, two_player_vtrace
from vec_env import py_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--games-per-step", type=int, default=4096)
    ap.add_argument("--sched-sizes", type=int, nargs="+", default=[50000])
    ap.add_argument("--policy-steps", type=int, default=4000, help="phase 1: policy only")
    ap.add_argument("--joint-steps", type=int, default=2000, help="phase 2: policy + value")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    cfg = Config(dice=1, games_per_step=args.games_per_step, sched_sizes=tuple(args.sched_sizes),
                 sched_repeats=(1,) * len(args.sched_sizes))
    agent = RNaD(cfg)
    net = agent.net
    target = torch.tensor(np.load(CFR_PATH).reshape(-1, TREE.A), dtype=torch.float32)
    static, win, wmask, mask = (torch.stack(x) for x in zip(*[py_features(s, SPEC) for s in states()]))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def policy_kl():
        logp = torch.log_softmax(net.policy_logits(static, win, wmask, mask).masked_fill(~mask, NEG), -1)
        return (target * (torch.log(target.clamp(min=1e-12)) - logp)).masked_fill(~mask, 0.0).sum(-1).mean()

    def nash_conv():
        with torch.no_grad():
            p = torch.softmax(net.policy_logits(static, win, wmask, mask).masked_fill(~mask, NEG), -1)
        p = p.double().numpy().reshape(TREE.H, 6, TREE.A)
        return TREE.nash_conv(p / p.sum(-1, keepdims=True))[0]

    t0 = time.time()
    for k in range(1, args.policy_steps + 1):
        kl = policy_kl()
        opt.zero_grad()
        kl.backward()
        opt.step()
        if k % 1000 == 0:
            print(f"policy {k}: KL {kl.item():.5f} NashConv {nash_conv():.4f} [{time.time() - t0:.0f}s]", flush=True)
    for k in range(1, args.joint_steps + 1):
        with torch.no_grad():
            traj = collect(agent.env, net)
            zero = torch.zeros(traj.S)
            ret, _ = two_player_vtrace(traj, zero, traj.logmu, traj.logmu, cfg.eta, pen=zero)   # Monte Carlo returns
        n_p = torch.bincount(traj.player, minlength=2).clamp(min=1).float()
        val = ((net.value(traj.static, traj.win, traj.wmask) - ret) ** 2 / n_p[traj.player]).sum()
        kl = policy_kl()
        opt.zero_grad()
        (kl + val).backward()
        opt.step()
        if k % 500 == 0:
            print(f"joint {k}: KL {kl.item():.5f} value loss {val.item():.4f} NashConv {nash_conv():.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    agent.target.load_state_dict(net.state_dict())
    agent.reg_cur = agent.reg_prev = NetReg(net)
    agent.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=(0.0, 0.999), eps=cfg.adam_eps)
    torch.save(agent.state_dict(), args.out)
    print(f"saved {args.out}: CFR+-initialized step-0 R-NaD checkpoint, NashConv {nash_conv():.4f}", flush=True)


if __name__ == "__main__":
    main()
