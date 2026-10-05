"""R-NaD (Regularized Nash Dynamics) with vectorized self-play, after DeepNash (Perolat et al., 2022).

One network (policy logits + value) plays both seats. Each learner step:
  1. self-play a batch of full games with the current policy (all games advance in lockstep),
  2. transform rewards with a penalty that pulls the policy toward a frozen regularization
     policy:  r' = r - eta*log(pi/pi_reg) for the acting player, +eta*log(pi/pi_reg) for the other,
  3. estimate values/Q with the paper's two-player v-trace over the whole game (no bootstrapping),
  4. update the policy with the NeuRD loss (a policy-gradient step on logits) and the value net
     by regression, then move a slow-moving target network toward the online network.
After `iter_steps` steps the target network's policy becomes the new regularization policy.
"""
import copy
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from game import Bot
from nfsp_model import NFSPConfig, action_to_bid
from rnad_net import InfoSetNet, RNaDNet
from vec_env import STATIC_DIM, Spec, VecEnv, py_features

NEG = float("-inf")


@dataclass
class Config:
    dice: int = 5
    window: int = 8
    gru_hidden: int = 64
    score_hidden: int = 64
    games_per_step: int = 1024
    iter_steps: int = 1000       # learner steps per R-NaD iteration (the paper's delta_m)
    eta: float = 0.2             # regularization strength
    lr: float = 3e-4
    gamma: float = 0.01          # target-network averaging rate
    beta: float = 2.0            # NeuRD logit threshold
    adv_clip: float = 1e4
    grad_clip: float = 10.0
    value_coef: float = 1.0
    adv_mode: str = "vtrace"    # "vtrace": paper's importance-weighted Q; "critic": learned per-action Q head
    q_coef: float = 1.0
    rho_bar: float = 1.0
    c_bar: float = 1.0
    chunk: int = 8192
    shaping: float = 0.0         # potential-based reward shaping on dice counts (0 = off, paper-faithful)
    # Options added to match DeepMind's reference implementation (open_spiel rnad.py). The defaults
    # keep the original behaviour so older checkpoints resume unchanged; train_rnad --preset reference
    # switches them on.
    net: str = "face"            # "face": RNaDNet; "infoset": MLP on the exact information set (reference)
    hidden: int = 256            # InfoSetNet torso width
    center: str = "legal"        # NeuRD logit centering: mean over "legal" actions, or over "all" actions (reference)
    adam_eps: float = 1e-8       # reference: 1e-7 (optax eps=10e-8)
    init: str = "torch"          # "torch" default init, or "haiku": truncated normal std 1/sqrt(fan_in), zero bias
    init_reg: str = "uniform"    # initial regularization policy: "uniform", or "net" = the initial network (reference)
    face_rank: bool = False      # RNaDNet: face-rank inputs to the shared per-face MLP
    info_ctx: int = 0            # RNaDNet: width of the exact-round-encoding context MLP (0 = off)
    reg_reward: str = "sampled"  # penalty in the reward stream: "sampled" log-ratio of the taken action (paper's
                                 # equations) or "expected" KL(pi || pi_reg) at each step (reference code)
    loss_norm: str = "global"    # "global": mean over all rows; "per_player": mean per player, summed (reference);
                                 # "infoset": policy loss averaged per distinct information set in the batch, so
                                 # every visited infoset takes the same-size step (value loss stays per player)
    sched_sizes: tuple = ()      # delta_m schedule as in the reference's EntropySchedule; empty -> (iter_steps,)
    sched_repeats: tuple = ()

    @property
    def spec(self) -> Spec:
        return Spec(dice=self.dice, window=self.window)


class EntropySchedule:
    """The reference implementation's schedule of regularization-policy updates.

    EntropySchedule([3, 5, 10], [2, 4, 1]) updates after steps [3, 6, 11, 16, 21, 26, 36, ...]:
    two iterations of size 3, four of size 5, then size 10 forever (the last repeat must be 1).
    """

    def __init__(self, sizes, repeats):
        if len(sizes) != len(repeats) or not sizes or any(r <= 0 for r in repeats) or repeats[-1] != 1:
            raise ValueError(f"bad entropy schedule: sizes={sizes} repeats={repeats}")
        sched = [0]
        for size, rep in zip(sizes, repeats):
            sched.extend([sched[-1] + (i + 1) * size for i in range(rep)])
        self.schedule = sched

    def iteration(self, step: int):
        """(index, start, size) of the R-NaD iteration that contains learner step `step` (0-based)."""
        sch = self.schedule
        if step >= sch[-1]:
            size = sch[-1] - sch[-2]
            k = (step - sch[-1]) // size
            return len(sch) - 1 + k, sch[-1] + k * size, size
        k = max(i for i, s in enumerate(sch) if s <= step)
        return k, sch[k], sch[k + 1] - sch[k]

    def __call__(self, step: int):
        """(alpha, update_after_this_step): alpha mixes the newest regularization policy with the one
        before (min(1, 2 n / size)); the regularization policy is replaced after the iteration's last step."""
        _, start, size = self.iteration(step)
        return min(1.0, 2.0 * (step - start) / size), step > 0 and step == start + size - 1


class Traj:
    """Flat rows of every decision of a batch of complete games, in time-major order."""

    def __init__(self, cols, offsets, winner, n_envs):
        for k, v in cols.items():
            setattr(self, k, v)
        self.offsets, self.winner, self.n_envs = offsets, winner, n_envs
        self.S = self.act.shape[0]


@torch.no_grad()
def collect(env: VecEnv, net: RNaDNet, shaping: float = 0.0) -> Traj:
    env.reset()
    active = torch.arange(env.n)
    keys = ("env", "static", "win", "wmask", "mask", "act", "logmu", "logmu_all", "player", "final", "shape")
    rec = {k: [] for k in keys}
    offsets, total = [0], 0
    winner = torch.full((env.n,), -1, dtype=torch.long)
    while active.numel():
        obs = env.observe(active)
        logits = net.policy_logits(obs.static, obs.win, obs.wmask, obs.mask).masked_fill(~obs.mask, NEG)
        logp = torch.log_softmax(logits, -1)
        a = torch.multinomial(logp.exp(), 1).squeeze(1)
        done, w, shape = env.step(active, a, shaping)
        for k, v in zip(keys, (active, obs.static, obs.win, obs.wmask, obs.mask, a,
                               logp.gather(1, a[:, None]).squeeze(1), logp.masked_fill(~obs.mask, 0.0),
                               obs.player, done, shape)):
            rec[k].append(v)
        winner[active[done]] = w[done]
        total += active.numel()
        offsets.append(total)
        active = active[~done]
    return Traj({k: torch.cat(v) for k, v in rec.items()}, offsets, winner, env.n)


def chunked(fn, chunk, *tensors):
    return torch.cat([fn(*[t[i: i + chunk] for t in tensors]) for i in range(0, tensors[0].shape[0], chunk)])


class UniformReg:
    def log_probs(self, static, win, wmask, mask):
        n = mask.sum(1, keepdim=True).clamp(min=1).float()
        return (-torch.log(n)).expand(mask.shape).masked_fill(~mask, 0.0)


class NetReg:
    def __init__(self, net):
        self.net = copy.deepcopy(net).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def log_probs(self, static, win, wmask, mask):
        logits = self.net.policy_logits(static, win, wmask, mask).masked_fill(~mask, NEG)
        return torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)


class NFSPReg:
    """Regularization policy taken from a trained NFSP shared_face model (its average-policy network).

    The paper allows any full-support policy as the initial regularization policy; starting from a
    good one makes the first R-NaD fixed points play well instead of climbing out of random play.
    The NFSP net reads the last K bids of the window (its own training saw the full history).
    """

    def __init__(self, path, spec):
        from nfsp_model import NFSPConfig, build_net
        ck = torch.load(path, map_location="cpu")
        cfg = NFSPConfig(**ck["config"])
        assert cfg.arch == "shared_face" and cfg.dice_count == spec.dice, "need a shared_face NFSP model for this dice count"
        self.path, self.net = path, build_net(cfg)
        self.net.load_state_dict(ck["sl_net"])
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def log_probs(self, static, win, wmask, mask):
        B, K, D = win.shape
        m = wmask.sum(1)
        j = torch.arange(K)[None, :]
        src = (j + (K - m)[:, None]).clamp(max=K - 1)                  # left-padded window -> right-padded sequence
        seq = win.gather(1, src[..., None].expand(-1, -1, D)) * (j < m[:, None])[..., None]
        logits = self.net(static, seq, m.clamp(min=1), mask).masked_fill(~mask, NEG)
        return torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)


def two_player_vtrace(traj, v, logpi_taken, logreg_taken, eta, ratio=None, rho_bar=1.0, c_bar=1.0, pen=None):
    """The paper's two-player v-trace, computed backward over whole games without bootstrapping.

    `pen` is the regularization penalty paid by the acting player at each row (and received by the other
    player); by default eta * log(pi/pi_reg) of the sampled action, as in the paper's equations. The
    reference code passes eta * KL(pi || pi_reg) instead, its expectation over actions (lower variance).
    Returns (vh_row, gm_row): the value target for each decision's acting player, and
    (return-to-go estimate - v) used to build the importance-weighted Q estimate.
    """
    S, n = traj.S, traj.n_envs
    if pen is None:
        pen = eta * (logpi_taken - logreg_taken)
    r_env = traj.shape.clone()
    fin = traj.final
    r_env[fin] += torch.where(traj.winner[traj.env[fin]] == traj.player[fin], 1.0, -1.0)
    r_act, r_oth = r_env - pen, -r_env + pen
    ratio = torch.ones(S) if ratio is None else ratio
    vh, Vn, rh, xi = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 2), torch.ones(n, 2)
    vh_row, gm_row = torch.zeros(S), torch.zeros(S)
    offs = traj.offsets
    for t in range(len(offs) - 2, -1, -1):
        sl = slice(offs[t], offs[t + 1])
        e, psi = traj.env[sl], traj.player[sl]
        oth, rt, vt = 1 - psi, ratio[sl], v[sl]
        vh_n, Vn_n, rh_n, xi_n = vh[e, psi], Vn[e, psi], rh[e, psi], xi[e, psi]
        w = rt * xi_n
        rho, c = w.clamp(max=rho_bar), w.clamp(max=c_bar)
        vh_new = vt + rho * (r_act[sl] + rt * rh_n + Vn_n - vt) + c * (vh_n - Vn_n)
        gm_row[sl] = r_env[sl] + rt * (rh_n + vh_n) - vt
        vh_row[sl] = vh_new
        vh[e, psi], Vn[e, psi], rh[e, psi], xi[e, psi] = vh_new, vt, 0.0, 1.0
        rh[e, oth] = r_oth[sl] + rt * rh[e, oth]
        xi[e, oth] = rt * xi[e, oth]
    return vh_row, gm_row


def build_net(cfg: Config):
    if cfg.net == "infoset":
        return InfoSetNet(cfg.spec, cfg.hidden)
    return RNaDNet(cfg.spec, cfg.gru_hidden, cfg.score_hidden, cfg.face_rank, cfg.info_ctx)


def vtrace_advantage(pi, pen, act, logmu, gm):
    """Advantage of each action from the paper's importance-weighted Q estimate (eq. 5):
    Q(a) = -pen(a) + 1[a = taken] / mu(taken) * gm, minus its expectation under pi."""
    onehot = F.one_hot(act, pi.shape[-1]).float()
    q_hat = -pen + onehot / logmu.exp()[:, None] * gm[:, None]
    return q_hat - (pi * q_hat).sum(-1, keepdim=True)


def neurd_loss(logits, mask, adv, beta, center="legal"):
    """Per-row NeuRD loss: push centered logits along the advantage, but only while they stay within
    [-beta, beta] of the center (the reference's apply_force_with_threshold). center="legal" uses the mean
    of the legal logits; "all" divides their sum by the number of actions, as the reference does."""
    n = mask.sum(-1, keepdim=True).float() if center == "legal" else float(mask.shape[-1])
    lc = logits - (logits * mask).sum(-1, keepdim=True) / n
    force = torch.where(adv > 0, lc < beta, lc > -beta)
    return -(mask * force * lc * adv).sum(-1)


def haiku_init_(net):
    """Haiku's default Linear init (the reference network's): truncated normal, std 1/sqrt(fan_in), zero bias."""
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            std = 1.0 / math.sqrt(m.in_features)
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            torch.nn.init.zeros_(m.bias)


class RNaD:
    def __init__(self, cfg: Config):
        self.cfg, self.spec = cfg, cfg.spec
        self.net = build_net(cfg)
        if cfg.init == "haiku":
            haiku_init_(self.net)
        self.target = copy.deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(0.0, 0.999), eps=cfg.adam_eps)
        self.env = VecEnv(self.spec, cfg.games_per_step)
        self.reg_cur, self.reg_prev = UniformReg(), UniformReg()
        if cfg.init_reg == "net":                              # reference: regularize toward the initial network
            self.reg_cur = self.reg_prev = NetReg(self.net)
        self.step_count, self.iter, self.n_in_iter = 0, 0, 0
        self.schedule = EntropySchedule(tuple(cfg.sched_sizes) or (cfg.iter_steps,), tuple(cfg.sched_repeats) or (1,))

    def distill(self, reg, steps: int, lr: float = 1e-3, games: int = 512, log=print) -> None:
        """Supervised warm start: fit the online policy to `reg` on states reached by playing `reg`
        (so the first R-NaD iteration starts with the online policy at the reference, not far away).
        The value/q heads are left to R-NaD; the target network is synced afterwards."""
        class _Play:
            policy_logits = staticmethod(lambda s, w, wm, m: reg.log_probs(s, w, wm, m))

        env = VecEnv(self.spec, games)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for k in range(1, steps + 1):
            traj = collect(env, _Play)
            perm = torch.randperm(traj.S)
            tot, n = 0.0, 0
            for i in range(0, traj.S, 4096):
                ix = perm[i: i + 4096]
                mask = traj.mask[ix]
                logits = self.net.policy_logits(traj.static[ix], traj.win[ix], traj.wmask[ix], mask)
                logp = torch.log_softmax(logits.masked_fill(~mask, NEG), -1)
                ref = reg.log_probs(traj.static[ix], traj.win[ix], traj.wmask[ix], mask)
                kl = (ref.exp() * (ref - logp)).masked_fill(~mask, 0.0).sum(-1).mean()
                opt.zero_grad()
                kl.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                opt.step()
                tot, n = tot + float(kl), n + 1
            if k % 25 == 0 or k == steps:
                log(f"  distill {k}/{steps}: KL(ref||net)={tot / n:.4f}")
        self.target.load_state_dict(self.net.state_dict())

    @property
    def alpha(self) -> float:
        return self.schedule(self.step_count)[0]

    def learn_step(self) -> dict:
        cfg, spec = self.cfg, self.spec
        t0 = time.time()
        traj = collect(self.env, self.net, cfg.shaping)
        t1 = time.time()
        S, A = traj.S, spec.n_actions

        v = chunked(lambda s, w, m: self.target.value(s, w, m), cfg.chunk, traj.static, traj.win, traj.wmask)
        regf = lambda reg: chunked(reg.log_probs, cfg.chunk, traj.static, traj.win, traj.wmask, traj.mask)
        logreg = regf(self.reg_cur)
        alpha, update_reg = self.schedule(self.step_count)
        if alpha < 1.0:
            logreg = alpha * logreg + (1.0 - alpha) * regf(self.reg_prev)
        logreg_taken = logreg.gather(1, traj.act[:, None]).squeeze(1)
        pen = None
        if cfg.reg_reward == "expected":                        # on-policy: the behaviour policy is pi itself
            mu = traj.logmu_all.exp() * traj.mask
            pen = cfg.eta * (mu * (traj.logmu_all - logreg)).sum(-1)
        vh, gm = two_player_vtrace(traj, v, traj.logmu, logreg_taken, cfg.eta, None, cfg.rho_bar, cfg.c_bar, pen)
        if cfg.loss_norm in ("per_player", "infoset"):          # mean over each player's steps, summed over players
            n_p = torch.bincount(traj.player, minlength=2).clamp(min=1).float()
            wrow = 1.0 / n_p[traj.player]
        else:
            wrow = torch.full((S,), 1.0 / S)
        wpol = wrow
        if cfg.loss_norm == "infoset":                          # each distinct infoset's rows share weight 1/n_infosets(player)
            key = torch.cat([traj.player[:, None].float(), traj.static[:, STATIC_DIM:]], 1)
            _, inv, cnt = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
            first = torch.zeros(cnt.shape[0], dtype=torch.long).scatter_(0, inv, traj.player)
            n_inf = torch.bincount(first, minlength=2).clamp(min=1).float()
            wpol = 1.0 / (cnt[inv].float() * n_inf[traj.player])

        self.opt.zero_grad()
        acc = dict(pol=0.0, val=0.0, kl=0.0, ent=0.0, adv=0.0)
        for i in range(0, S, cfg.chunk):
            sl = slice(i, min(S, i + cfg.chunk))
            mask = traj.mask[sl]
            parts = self.net.parts(traj.static[sl], traj.win[sl], traj.wmask[sl])
            logits, qv = self.net.heads(traj.static[sl], traj.win[sl], traj.wmask[sl], mask, parts)
            value = self.net.value(traj.static[sl], traj.win[sl], traj.wmask[sl], parts)
            logp = torch.log_softmax(logits.masked_fill(~mask, NEG), -1)
            with torch.no_grad():
                pi = logp.exp()
                logpi = logp.masked_fill(~mask, 0.0)
                pen = cfg.eta * (logpi - logreg[sl])
                if cfg.adv_mode == "critic":
                    qd = qv.detach() * mask
                    adv = (qd - (pi * qd).sum(-1, keepdim=True)) - (pen - (pi * pen).sum(-1, keepdim=True))
                else:
                    adv = vtrace_advantage(pi, pen, traj.act[sl], traj.logmu[sl], gm[sl])
                adv = adv.clamp(-cfg.adv_clip, cfg.adv_clip)
            nl = mask.sum(-1, keepdim=True).float()
            pol_loss = neurd_loss(logits, mask, adv, cfg.beta, cfg.center)
            val_loss = 0.5 * (value - vh[sl]) ** 2
            q_loss = 0.5 * (qv.gather(1, traj.act[sl][:, None]).squeeze(1) - (gm[sl] + v[sl])) ** 2 \
                if cfg.adv_mode == "critic" else torch.zeros_like(val_loss)
            (wpol[sl] * pol_loss + wrow[sl] * (cfg.value_coef * val_loss + cfg.q_coef * q_loss)).sum().backward()
            with torch.no_grad():
                acc["pol"] += float(pol_loss.sum())
                acc["val"] += float(val_loss.sum())
                acc["kl"] += float((pi * (logpi - logreg[sl])).sum())
                acc["ent"] += float(-(pi * logpi).sum())
                acc["adv"] += float((adv.abs() * mask).sum() / nl.mean())
        if cfg.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        else:
            gn = torch.norm(torch.stack([p.grad.norm() for p in self.net.parameters() if p.grad is not None]))
        self.opt.step()
        with torch.no_grad():
            for pt, pn in zip(self.target.parameters(), self.net.parameters()):
                pt.mul_(1.0 - cfg.gamma).add_(pn, alpha=cfg.gamma)

        if update_reg:
            self.reg_prev, self.reg_cur = self.reg_cur, NetReg(self.target)
        self.step_count += 1
        self.iter, start, _ = self.schedule.iteration(self.step_count)
        self.n_in_iter = self.step_count - start
        return dict(rows=S, games=traj.n_envs, len=S / traj.n_envs, kl=acc["kl"] / S, ent=acc["ent"] / S,
                    val=acc["val"] / S, pol=acc["pol"] / S, gn=float(gn), t_roll=t1 - t0, t_learn=time.time() - t1,
                    seat0=float((traj.winner == 0).float().mean()))

    def state_dict(self):
        return dict(cfg=asdict(self.cfg), net=self.net.state_dict(), target=self.target.state_dict(),
                    opt=self.opt.state_dict(), step=self.step_count, iter=self.iter, n_in_iter=self.n_in_iter,
                    reg_cur=self._reg_state(self.reg_cur), reg_prev=self._reg_state(self.reg_prev))

    @staticmethod
    def _reg_state(reg):
        if isinstance(reg, UniformReg):
            return None
        if isinstance(reg, NFSPReg):
            return {"nfsp_path": reg.path}
        return {"net": reg.net.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])
        self.step_count, self.iter, self.n_in_iter = sd["step"], sd["iter"], sd["n_in_iter"]
        for name in ("reg_cur", "reg_prev"):
            st = sd[name]
            if st is None:
                setattr(self, name, UniformReg())
            elif "nfsp_path" in st:
                setattr(self, name, NFSPReg(st["nfsp_path"], self.spec))
            else:
                net = copy.deepcopy(self.net)
                net.load_state_dict(st["net"])
                setattr(self, name, NetReg(net))


def policy_probs(net, static, win, wmask, mask, threshold=0.0):
    """Deployed policy: softmax over legal actions, optionally dropping tiny-probability actions."""
    with torch.no_grad():
        p = torch.softmax(net.policy_logits(static, win, wmask, mask).masked_fill(~mask, NEG), -1)
        if threshold > 0:
            kept = p * ((p >= threshold) & mask)
            tot = kept.sum(-1, keepdim=True)
            p = torch.where(tot > 0, kept / tot.clamp(min=1e-12), p)
    return p


class RNaDBot(Bot):
    """Plays a trained R-NaD network in the Python engine (evaluation vs the heuristic bots)."""
    name = "RNaD"

    def __init__(self, net, spec: Spec, threshold: float = 0.0, sample: bool = True):
        self.net, self.spec, self.threshold, self.sample = net, spec, threshold, sample
        self.cfg = NFSPConfig(dice_count=spec.dice)

    def act(self, state):
        static, win, wmask, mask = py_features(state, self.spec)
        p = policy_probs(self.net, static[None], win[None], wmask[None], mask[None], self.threshold)[0]
        idx = int(torch.multinomial(p, 1)) if self.sample else int(p.argmax())
        return action_to_bid(idx, self.cfg)


def load_policy_net(path, which="target"):
    """Load the deployable network from a checkpoint saved by train_rnad.py."""
    sd = torch.load(path, map_location="cpu")
    cfg = Config(**sd["cfg"])
    net = build_net(cfg)
    net.load_state_dict(sd[which])
    net.eval()
    return net, cfg
