"""Train a Liar's Dice bot with R-NaD and vectorized self-play.

Examples:
    python3 train_rnad.py --dice 1 --total-steps 6000            # small game, prints exact NashConv
    python3 train_rnad.py --dice 5 --total-steps 30000 --max-hours 8   # full game, evals vs SmartBot/BestBot

Checkpoints go to checkpoints/rnad_d<dice>/ (rnad_latest.pt plus numbered ones). The deployable
policy is the slow-moving `target` network. Use --resume to continue a run.
"""
import argparse
import os
import random
import time

import torch

from game import Game, PlayerState
from rnad import Config, NFSPReg, RNaD, RNaDBot, config_from_dict


def eval_vs_bot(net, spec, make_opp, n_games, threshold):
    wins = 0
    for g in range(n_games):
        players = [PlayerState("RNaD", bot=RNaDBot(net, spec, threshold)), PlayerState("Opp", bot=make_opp())]
        if g % 2:
            players.reverse()
        winner = Game(players, starting_dice=spec.dice).play()
        wins += players[winner].name == "RNaD"
    return wins / n_games


def main():
    d = Config()                       # defaults match DeepMind's reference implementation (see rnad.Config)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dice", type=int, default=d.dice)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--games-per-step", type=int, default=d.games_per_step)
    ap.add_argument("--iter-steps", type=int, default=d.iter_steps, help="learner steps per R-NaD iteration (delta_m)")
    ap.add_argument("--sched-sizes", type=int, nargs="+", default=None,
                    help="delta_m schedule (reference EntropySchedule sizes); overrides --iter-steps")
    ap.add_argument("--sched-repeats", type=int, nargs="+", default=None, help="parallel to --sched-sizes (last must be 1)")
    ap.add_argument("--eta", type=float, default=d.eta)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lr-points", type=str, default="",
                    help="lr schedule 'STEP:LR,STEP:LR', linear from (0, --lr), constant after the last point")
    ap.add_argument("--gamma", type=float, default=d.gamma, help="target-network averaging rate")
    ap.add_argument("--beta", type=float, default=d.beta, help="NeuRD logit threshold")
    ap.add_argument("--adv-clip", type=float, default=d.adv_clip, help="clip on the importance-weighted advantage")
    ap.add_argument("--grad-clip", type=float, default=d.grad_clip, help="global grad-norm clip before Adam (0 = off)")
    ap.add_argument("--value-coef", type=float, default=d.value_coef, help="weight of (v - target)^2")
    ap.add_argument("--adam-eps", type=float, default=d.adam_eps)
    ap.add_argument("--net", type=str, default=d.net, choices=["infoset", "face"],
                    help="infoset: MLP on the exact information set (reference); face: face-shared RNaDNet")
    ap.add_argument("--hidden", type=int, default=d.hidden, help="InfoSetNet torso width")
    ap.add_argument("--score-hidden", type=int, default=d.score_hidden, help="face net: per-action scoring MLP width")
    ap.add_argument("--face-rank", action="store_true", help="face net: give the shared face MLP each face's rank")
    ap.add_argument("--info-ctx", type=int, default=d.info_ctx, help="face net: width of the exact-round context MLP")
    ap.add_argument("--init", type=str, default=d.init, choices=["haiku", "torch"])
    ap.add_argument("--init-reg", type=str, default=d.init_reg, choices=["net", "uniform"])
    ap.add_argument("--reg-reward", type=str, default=d.reg_reward, choices=["expected", "sampled"],
                    help="regularization penalty in the reward stream: expected KL (reference) or sampled log-ratio")
    ap.add_argument("--loss-norm", type=str, default=d.loss_norm, choices=["per_player", "global", "infoset"])
    ap.add_argument("--adv-mode", type=str, default=d.adv_mode, choices=["vtrace", "critic"])
    ap.add_argument("--q-coef", type=float, default=d.q_coef)
    ap.add_argument("--shaping", type=float, default=d.shaping, help="potential-based shaping strength (0 = off)")
    ap.add_argument("--chunk", type=int, default=d.chunk)
    ap.add_argument("--preset", type=str, default="none", choices=["none", "reference"],
                    help="deprecated: the defaults now match the reference, so this does nothing")
    ap.add_argument("--finetune-thr", type=float, default=0.03,
                    help="dice 1: also report NashConv after DeepNash's fine-tuning (threshold + discretization)")
    ap.add_argument("--finetune-disc", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-games", type=int, default=500, help="games per opponent at each eval (dice 5 only)")
    ap.add_argument("--threshold", type=float, default=0.0, help="drop actions below this probability at eval time")
    ap.add_argument("--init-reg-nfsp", type=str, default=None,
                    help="use this NFSP shared_face checkpoint as the initial regularization policy")
    ap.add_argument("--distill-steps", type=int, default=200,
                    help="supervised warm-start steps onto the NFSP regularization policy")
    ap.add_argument("--eval-nfsp", type=str, default=None,
                    help="optional NFSP checkpoint to also play head-to-head at each eval (dice 5 only)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop (and save) after this much wall time")
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    sizes = tuple(args.sched_sizes or ())
    repeats = tuple(args.sched_repeats or ((1,) * len(sizes)))
    lr_points = tuple((int(p.split(":")[0]), float(p.split(":")[1])) for p in args.lr_points.split(",") if p)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ckpt_dir = args.checkpoint_dir or os.path.join("checkpoints", f"rnad_d{args.dice}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:                    # rebuild exactly the saved run (architecture and options)
        sd = torch.load(args.resume, map_location="cpu")
        cfg = config_from_dict(sd["cfg"])
        agent = RNaD(cfg)
        agent.load_state_dict(sd)
        print(f"resumed from {args.resume} at step {agent.step_count} (iter {agent.iter})", flush=True)
    else:
        cfg = Config(dice=args.dice, games_per_step=args.games_per_step, iter_steps=args.iter_steps,
                     sched_sizes=sizes, sched_repeats=repeats, eta=args.eta, lr=args.lr, lr_points=lr_points,
                     gamma=args.gamma, beta=args.beta, adv_clip=args.adv_clip, grad_clip=args.grad_clip,
                     value_coef=args.value_coef, adam_eps=args.adam_eps, net=args.net, hidden=args.hidden,
                     score_hidden=args.score_hidden, face_rank=args.face_rank, info_ctx=args.info_ctx,
                     init=args.init, init_reg=args.init_reg, reg_reward=args.reg_reward, loss_norm=args.loss_norm,
                     adv_mode=args.adv_mode, q_coef=args.q_coef, shaping=args.shaping, chunk=args.chunk)
        agent = RNaD(cfg)
    spec = cfg.spec
    if args.init_reg_nfsp and not args.resume:
        agent.reg_cur = agent.reg_prev = NFSPReg(args.init_reg_nfsp, spec)
        print(f"initial regularization policy: NFSP model {args.init_reg_nfsp}", flush=True)
    print(f"cfg={cfg}\ncheckpoint_dir={ckpt_dir} total_steps={args.total_steps}", flush=True)

    def save(tag):
        sd = agent.state_dict()
        torch.save(sd, os.path.join(ckpt_dir, "rnad_latest.pt"))
        if tag:
            torch.save(sd, os.path.join(ckpt_dir, f"rnad_step{tag}.pt"))

    exact = {}

    def evaluate():
        if spec.dice == 1:                                     # exact NashConv (exact_d1), raw and fine-tuned
            from exact_d1 import Tree, discretize_policy, threshold_policy
            from exploit_d1 import make_state, net_prob_fn
            if not exact:
                exact["tree"] = tree = Tree(spec.qmax)
                exact["states"] = [make_state(h, d) for h in tree.hists for d in range(1, 7)]
            tree = exact["tree"]
            pi = net_prob_fn(agent.target, spec, args.threshold)(exact["states"]).reshape(tree.H, 6, tree.A)
            pi = pi.astype("float64") / pi.sum(-1, keepdims=True)
            out = f"NashConv={tree.nash_conv(pi)[0]:.4f}"
            if args.finetune_thr > 0 or args.finetune_disc > 0:
                ft = discretize_policy(threshold_policy(pi, tree.legal, args.finetune_thr), args.finetune_disc)
                out += f" NashConv_finetuned={tree.nash_conv(ft)[0]:.4f}"
            return out
        from bots import SmartBot
        from best_bot import build_best_bot
        ws = eval_vs_bot(agent.target, spec, SmartBot, args.eval_games, args.threshold)
        wb = eval_vs_bot(agent.target, spec, build_best_bot, args.eval_games, args.threshold)
        out = f"win_vs_SmartBot={ws:.1%} win_vs_BestBot={wb:.1%}"
        if args.eval_nfsp:
            from nfsp_model import NFSPBot
            wn = eval_vs_bot(agent.target, spec, lambda: NFSPBot.from_checkpoint(args.eval_nfsp), args.eval_games,
                             args.threshold)
            out += f" win_vs_NFSP={wn:.1%}"
        return out

    if args.init_reg_nfsp and not args.resume and args.distill_steps:
        t0 = time.time()
        agent.distill(agent.reg_cur, args.distill_steps, log=lambda s: print(s, flush=True))
        print(f"distillation done in {time.time() - t0:.0f}s; distilled policy: {evaluate()}", flush=True)

    t_start, hist = time.time(), []
    try:
        while agent.step_count < args.total_steps:
            m = agent.learn_step()
            hist.append(m)
            s = agent.step_count
            if s % args.log_every == 0 or s == args.total_steps:
                avg = lambda k: sum(h[k] for h in hist) / len(hist)
                el = (time.time() - t_start) / 60
                print(f"[step {s}/{args.total_steps} iter {agent.iter}+{agent.n_in_iter}/{agent.schedule.iteration(s)[2]}] "
                      f"len={avg('len'):.1f} kl={avg('kl'):.4f} ent={avg('ent'):.3f} val={avg('val'):.4f} "
                      f"gn={avg('gn'):.2f} lr={hist[-1]['lr']:.2e} seat0={avg('seat0'):.3f} roll={avg('t_roll'):.2f}s learn={avg('t_learn'):.2f}s "
                      f"elapsed={el:.1f}m", flush=True)
                hist = []
            if s % args.eval_every == 0 or s == args.total_steps:
                t0 = time.time()
                res = evaluate()
                print(f"  >> eval @ step {s} (iter {agent.iter}): {res} [{time.time() - t0:.0f}s]", flush=True)
                save(s)
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                print(f"max-hours reached at step {s}; saving and stopping", flush=True)
                save(s)
                break
    except KeyboardInterrupt:
        print("interrupted; saving", flush=True)
        save(agent.step_count)
        raise
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
