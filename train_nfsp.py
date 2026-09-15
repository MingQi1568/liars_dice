"""Train a 1v1 Liar's Dice bot via NFSP (Neural Fictitious Self-Play).

One shared Q-network + one shared SL-network play both seats (state is always
canonicalized to "me" vs "opponent", so a single symmetric agent controls both sides).
Each episode = one full game (possibly many rounds) per game.py's actual win condition
(first to 0 dice wins); reward is 0 until the terminal step, then +-1.

--total-episodes is the target *cumulative* episode count (not "how many more to run") --
epsilon/LR decay schedules are computed as a fraction of this total, so it applies whether
you're starting fresh or resuming (extending the total horizon re-plans the remaining decay
over the new, longer horizon rather than leaving epsilon/LR stuck at their old floor).

Usage:
    python3 train_nfsp.py --total-episodes 50000 --dice 5
    python3 train_nfsp.py --resume checkpoints/nfsp_latest.pt --total-episodes 250000
"""
import argparse
import os
import random
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import optim

from game import Game, PlayerState
from bots import SmartBot
from nfsp_model import NFSPConfig, NFSPNet, NFSPTrainingBot, NFSPBot, collate_states
from nfsp_buffers import ReplayBuffer, ReservoirBuffer, Transition


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(q_net, sl_net, cfg, eta, epsilon, device):
    mode_a = "br" if random.random() < eta else "avg"
    mode_b = "br" if random.random() < eta else "avg"
    bot_a = NFSPTrainingBot(q_net, sl_net, cfg, mode_a, epsilon, device)
    bot_b = NFSPTrainingBot(q_net, sl_net, cfg, mode_b, epsilon, device)
    players = [PlayerState("P0", bot=bot_a), PlayerState("P1", bot=bot_b)]
    game = Game(players, starting_dice=cfg.dice_count, verbose=False)
    winner = game.play()

    transitions = []
    sl_samples = []
    for i, bot in enumerate((bot_a, bot_b)):
        reward = 1.0 if i == winner else -1.0
        traj = bot.trajectory
        n = len(traj)
        for t in range(n):
            feats, action_idx, _mask = traj[t]
            done = t == n - 1
            r = reward if done else 0.0
            next_state = traj[t + 1][0] if not done else None
            next_mask = traj[t + 1][2] if not done else None
            transitions.append(Transition(feats, action_idx, r, next_state, done, next_mask))
        sl_samples.extend(bot.sl_samples)
    return transitions, sl_samples


def train_q_step(q_net, q_target, optimizer, batch, gamma, device, cfg):
    actions = torch.tensor([t.action for t in batch], device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=device)

    states = [t.state for t in batch]
    hand, dice, seq, lengths = collate_states(states, device)
    q_values = q_net(hand, dice, seq, lengths)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Terminal transitions have no next_state; fill with the current state as a dummy --
    # its Q-value gets multiplied by 0 via `dones` below, so it never contributes.
    next_states = [t.next_state if t.next_state is not None else t.state for t in batch]
    next_masks = torch.stack([
        t.next_mask if t.next_mask is not None else torch.zeros(cfg.n_actions, dtype=torch.bool)
        for t in batch
    ]).to(device)
    ns_hand, ns_dice, ns_seq, ns_lengths = collate_states(next_states, device)
    with torch.no_grad():
        next_q_online = q_net(ns_hand, ns_dice, ns_seq, ns_lengths)
        next_q_online = next_q_online.masked_fill(~next_masks, float("-inf"))
        next_actions = next_q_online.argmax(dim=1)
        next_q_target = q_target(ns_hand, ns_dice, ns_seq, ns_lengths)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        next_q = torch.where(dones, torch.zeros_like(next_q), next_q)
        target = rewards + gamma * next_q

    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_sl_step(sl_net, optimizer, batch, device):
    states = [b[0] for b in batch]
    actions = torch.tensor([b[1] for b in batch], device=device)
    hand, dice, seq, lengths = collate_states(states, device)
    logits = sl_net(hand, dice, seq, lengths)
    loss = F.cross_entropy(logits, actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(sl_net, cfg, device, opponent_builder, n_games):
    wins = 0
    for g in range(n_games):
        eval_bot = NFSPBot(sl_net, cfg, device=device, sample=True)
        opp = opponent_builder()
        players = [PlayerState("NFSP", bot=eval_bot), PlayerState("Opp", bot=opp)]
        if g % 2 == 1:
            players = list(reversed(players))
        game = Game(players, starting_dice=cfg.dice_count, verbose=False)
        winner = game.play()
        if players[winner].name == "NFSP":
            wins += 1
    return wins / n_games


def _best_bot_factory():
    from best_bot import build_best_bot
    return build_best_bot


def linear_decay(start, end, frac_done):
    return start - (start - end) * min(1.0, frac_done)


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value):
        self.total += value
        self.count += 1

    def pop_mean(self):
        mean = self.total / self.count if self.count else float("nan")
        self.total = 0.0
        self.count = 0
        return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-episodes", type=int, default=50_000,
                         help="target cumulative episode count (decay schedules are planned against this)")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--start-episode", type=int, default=None,
                         help="override the episode counter on resume (needed for checkpoints saved before "
                              "the 'episode' field was added to the checkpoint format)")
    parser.add_argument("--dice", type=int, default=5, help="ignored when --resume is set (uses checkpoint's config)")
    parser.add_argument("--eta", type=float, default=0.1, help="anticipatory param: P(play in best-response mode)")
    parser.add_argument("--eps-start", type=float, default=0.08)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay-frac", type=float, default=0.8, help="fraction of --total-episodes over which eps decays")
    parser.add_argument("--rl-buffer-size", type=int, default=100_000)
    parser.add_argument("--sl-buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rl-lr", type=float, default=1e-4, help="RL (Q-net) starting learning rate")
    parser.add_argument("--rl-lr-end", type=float, default=1e-5, help="RL (Q-net) ending learning rate")
    parser.add_argument("--sl-lr", type=float, default=1e-3, help="SL (avg-policy net) starting learning rate")
    parser.add_argument("--sl-lr-end", type=float, default=1e-4, help="SL (avg-policy net) ending learning rate")
    parser.add_argument("--lr-decay-frac", type=float, default=0.8, help="fraction of --total-episodes over which LR decays")
    parser.add_argument("--updates-per-episode", type=int, default=2, help="gradient steps per network, per episode")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--target-sync-every", type=int, default=500, help="episodes between target-net syncs")
    parser.add_argument("--eval-every", type=int, default=2000, help="episodes between eval/checkpoint/loss print")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = pick_device(args.device)

    start_ep = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        cfg = NFSPConfig(**ckpt["config"])
        start_ep = args.start_episode if args.start_episode is not None else ckpt.get("episode", 0)
        print(f"resuming from {args.resume} at episode {start_ep}", flush=True)
    else:
        ckpt = None
        cfg = NFSPConfig(dice_count=args.dice)

    print(f"device={device} dice_count={cfg.dice_count} n_actions={cfg.n_actions} "
          f"start_ep={start_ep} total_episodes={args.total_episodes}", flush=True)

    q_net = NFSPNet(cfg).to(device)
    q_target = NFSPNet(cfg).to(device)
    sl_net = NFSPNet(cfg).to(device)
    q_opt = optim.Adam(q_net.parameters(), lr=args.rl_lr)
    sl_opt = optim.Adam(sl_net.parameters(), lr=args.sl_lr)

    if ckpt is not None:
        q_net.load_state_dict(ckpt["q_net"])
        sl_net.load_state_dict(ckpt["sl_net"])
        if "q_opt" in ckpt:
            q_opt.load_state_dict(ckpt["q_opt"])
        if "sl_opt" in ckpt:
            sl_opt.load_state_dict(ckpt["sl_opt"])
        # Replay/reservoir buffers are intentionally NOT persisted -- they refill within a
        # few thousand fresh episodes and aren't worth the checkpoint-size/complexity cost.
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    rl_buffer = ReplayBuffer(args.rl_buffer_size)
    sl_buffer = ReservoirBuffer(args.sl_buffer_size)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    eps_decay_episodes = max(1, int(args.total_episodes * args.eps_decay_frac))
    lr_decay_episodes = max(1, int(args.total_episodes * args.lr_decay_frac))
    start_time = time.time()
    q_loss_tracker = RunningMean()
    sl_loss_tracker = RunningMean()

    def save_checkpoint(ep):
        payload = {
            "q_net": q_net.state_dict(), "sl_net": sl_net.state_dict(),
            "q_opt": q_opt.state_dict(), "sl_opt": sl_opt.state_dict(),
            "config": asdict(cfg), "episode": ep,
        }
        torch.save(payload, os.path.join(args.checkpoint_dir, "nfsp_latest.pt"))
        torch.save(payload, os.path.join(args.checkpoint_dir, f"nfsp_ep{ep}.pt"))

    ep = start_ep
    try:
        for ep in range(start_ep + 1, args.total_episodes + 1):
            epsilon = max(args.eps_end, linear_decay(args.eps_start, args.eps_end, ep / eps_decay_episodes))
            rl_lr = max(args.rl_lr_end, linear_decay(args.rl_lr, args.rl_lr_end, ep / lr_decay_episodes))
            sl_lr = max(args.sl_lr_end, linear_decay(args.sl_lr, args.sl_lr_end, ep / lr_decay_episodes))
            for g in q_opt.param_groups:
                g["lr"] = rl_lr
            for g in sl_opt.param_groups:
                g["lr"] = sl_lr

            transitions, sl_samples = run_episode(q_net, sl_net, cfg, args.eta, epsilon, device)
            for t in transitions:
                rl_buffer.push(t)
            for s in sl_samples:
                sl_buffer.push(s)

            for _ in range(args.updates_per_episode):
                if len(rl_buffer) >= args.batch_size:
                    q_loss = train_q_step(q_net, q_target, q_opt, rl_buffer.sample(args.batch_size), args.gamma, device, cfg)
                    q_loss_tracker.add(q_loss)
                if len(sl_buffer) >= args.batch_size:
                    sl_loss = train_sl_step(sl_net, sl_opt, sl_buffer.sample(args.batch_size), device)
                    sl_loss_tracker.add(sl_loss)

            if ep % args.target_sync_every == 0:
                q_target.load_state_dict(q_net.state_dict())

            if ep % args.eval_every == 0 or ep == args.total_episodes:
                wr_smart = evaluate(sl_net, cfg, device, lambda: SmartBot(), args.eval_games)
                wr_best = evaluate(sl_net, cfg, device, _best_bot_factory(), args.eval_games)
                elapsed = (time.time() - start_time) / 60
                print(
                    f"[ep {ep}/{args.total_episodes}] eps={epsilon:.3f} rl_lr={rl_lr:.2e} sl_lr={sl_lr:.2e} "
                    f"rl_buf={len(rl_buffer)} sl_buf={len(sl_buffer)} "
                    f"q_loss={q_loss_tracker.pop_mean():.4f} sl_loss={sl_loss_tracker.pop_mean():.4f} "
                    f"win_vs_SmartBot={wr_smart:.1%} win_vs_BestBot={wr_best:.1%} elapsed={elapsed:.1f}m",
                    flush=True,
                )
                save_checkpoint(ep)
    except KeyboardInterrupt:
        print("Interrupted -- saving checkpoint before exit.", flush=True)
        save_checkpoint(ep)
        raise

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
