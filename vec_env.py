"""Vectorized 1v1 Liar's Dice: N games advance in lockstep with tensor ops.

Rules mirror game.py (1s wild for faces 2-6, asymmetric 1s raise rules, the challenge
WINNER loses a die, first player to 0 dice wins, the challenge loser opens the next round).
Features mirror nfsp_model.encode_state_shared (static vector) plus a fixed window of the
last K bids encoded relative to the acting player's own hand, so the exact same inputs can
be produced from a Python GameState (see py_features) for bots and exact-exploitability tools.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nfsp_model import S_CLAIM, S_FACE1, S_FACES, S_GLOBAL, S_LIAR, STATIC_DIM  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class Spec:
    dice: int = 5
    window: int = 8

    @property
    def qmax(self) -> int:
        return 2 * self.dice

    @property
    def n_raise(self) -> int:
        return self.qmax * 6

    @property
    def n_actions(self) -> int:
        return self.n_raise + 1

    @property
    def bid_dim(self) -> int:
        return self.qmax + 7

    @property
    def max_hist(self) -> int:
        return 6 * self.qmax + 4

    @property
    def info_dim(self) -> int:
        """Width of the exact per-round information-set encoding appended after the STATIC_DIM features."""
        return 6 + 2 + self.n_raise + 1


def legal_mask(spec: Spec, cq: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
    """Legal actions [n, n_actions] given the standing bid (cf == 0 means no bid yet)."""
    n = cq.shape[0]
    q = torch.arange(1, spec.qmax + 1, device=cq.device)[None, :, None]
    f = torch.arange(1, 7, device=cq.device)[None, None, :]
    cqq, cff = cq[:, None, None], cf[:, None, None]
    none = cff == 0
    new_one, cur_one = f == 1, cff == 1
    standard = (q > cqq) | ((q == cqq) & (f > cff))
    gt = torch.where(new_one, q > cqq, torch.where(cur_one, q >= cqq, standard))
    raise_ok = none | gt
    liar_ok = ~none[:, 0, 0]
    return torch.cat([raise_ok.reshape(n, -1), liar_ok[:, None]], dim=1)


def info_features(spec, counts, hand, opp, hq, hf, hlen, opener, p):
    """Exact information set of the current round: own dice per face, both dice counts, the set of bids
    made this round (bids strictly increase, so the set fixes their order; who made each follows from
    the opener), and whether I opened. [n, spec.info_dim]."""
    n, dc = counts.shape[0], float(spec.dice)
    H = hq.shape[1]
    ar = torch.arange(H, device=counts.device)[None, :]
    ok = (ar < hlen[:, None]) & (hq >= 1) & (hq <= spec.qmax) & (hf >= 1)
    idx = ((hq - 1).clamp(0, spec.qmax - 1) * 6 + (hf - 1).clamp(0, 5)) * ok
    made = torch.zeros(n, spec.n_raise, device=counts.device).scatter_add_(1, idx, ok.float()).clamp(max=1.0)
    return torch.cat([counts[:, 1:7].float() / dc, (hand.float() / dc)[:, None], (opp.float() / dc)[:, None],
                      made, (opener == p).float()[:, None]], dim=1)


def make_features(spec, counts, hand, opp, cq, cf, hq, hf, hlen, opener, p, mask):
    """counts [n,7] (index = face 1..6); hand/opp/cq/cf/hlen/opener/p [n]; hq/hf [n,H]; mask [n,A].

    Returns static [n, STATIC_DIM + info_dim], win [n,K,bid_dim], wmask [n,K] (True where a real bid
    sits). Columns past STATIC_DIM hold info_features; the face-shared nets only read the first STATIC_DIM.
    """
    n = counts.shape[0]
    dev = counts.device
    dc, qmax = float(spec.dice), spec.qmax
    qcap = qmax + 1
    has = cf > 0
    c1 = counts[:, 1]
    eff = counts + c1[:, None]
    eff[:, 1] = c1

    H = hq.shape[1]
    ar = torch.arange(H, device=dev)[None, :]
    hvalid = ar < hlen[:, None]
    bidder = opener[:, None] ^ (ar & 1)
    by_opp = hvalid & (bidder != p[:, None])
    oh = F.one_hot(hf, 7).bool() & by_opp[:, :, None]
    opp_times = oh.sum(1)
    opp_max = (oh * hq[:, :, None]).amax(1)

    legal = mask[:, : spec.n_raise].reshape(n, qmax, 6)
    has_any = legal.any(1)
    first = legal.float().argmax(1)
    mn = torch.where(has_any, first + 1, torch.full_like(first, qcap))

    handf, oppf = hand.float(), opp.float()
    total = handf + oppf
    fcol = torch.arange(2, 7, device=dev)[None, :]
    cfc, hasc, cb_one = cf[:, None], has[:, None], (cf == 1)[:, None]
    above = (hasc & (fcol > cfc)).float()
    below = (hasc & ~cb_one & (fcol < cfc)).float()
    equal = (hasc & ~cb_one & (fcol == cfc)).float()
    eff_f = eff[:, 2:7].float()
    mn_f = mn[:, 1:6].float()
    faces = torch.stack([
        eff_f / dc,
        (handf / dc)[:, None].expand(n, 5),
        (total / (2 * dc))[:, None].expand(n, 5),
        above, below, equal,
        opp_max[:, 2:7].clamp(max=qcap).float() / qcap,
        opp_times[:, 2:7].clamp(max=qmax).float() / qmax,
        mn_f / qcap,
        (mn_f - eff_f) / qcap,
    ], dim=-1)

    c1f = c1.float()
    face1 = torch.stack([
        c1f / dc, handf / dc, total / (2 * dc), (has & (cf == 1)).float(),
        opp_max[:, 1].clamp(max=qcap).float() / qcap,
        opp_times[:, 1].clamp(max=qmax).float() / qmax,
        mn[:, 0].float() / qcap, (mn[:, 0].float() - c1f) / qcap,
    ], dim=-1)

    cfs = cf.clamp(min=1)
    eff_claim = eff.gather(1, cfs[:, None]).squeeze(1)
    needed = (cq - eff_claim).clamp(min=0)
    hasf = has.float()
    liar = torch.stack([
        needed.clamp(max=qcap).float() / qcap,
        (needed.float() / oppf.clamp(min=1.0)).clamp(max=2.0) / 2.0,
        eff_claim.float() / dc, total / (2 * dc), oppf / dc,
        cq.clamp(max=qcap).float() / qcap,
        (needed.float() > oppf).float(), (cf == 1).float(),
    ], dim=-1) * hasf[:, None]
    claim = F.one_hot(cfs - 1, 6).float() * hasf[:, None]
    glob = torch.stack([handf / dc, oppf / dc, c1f / dc, hasf], dim=-1)
    static = torch.cat([glob, faces.reshape(n, 50), face1, liar, claim,
                        info_features(spec, counts, hand, opp, hq, hf, hlen, opener, p)], dim=-1)

    K = spec.window
    j = torch.arange(K, device=dev)[None, :]
    idx = hlen[:, None] - K + j
    wvalid = idx >= 0
    safe = idx.clamp(min=0)
    qj, fj = hq.gather(1, safe), hf.gather(1, safe)
    prev_valid = idx >= 1
    fprev = hf.gather(1, (idx - 1).clamp(min=0))
    flag = torch.where((opener[:, None] ^ (safe & 1)) == p[:, None], 1.0, -1.0)
    eff_j = eff.gather(1, fj.clamp(min=1))
    needed_j = (qj - eff_j).clamp(min=0).clamp(max=qcap).float() / qcap
    same = (prev_valid & (fj == fprev)).float()
    nz = prev_valid & (fj != 1) & (fprev != 1)
    up, down = (nz & (fj > fprev)).float(), (nz & (fj < fprev)).float()
    qoh = F.one_hot((qj - 1).clamp(0, qmax - 1), qmax).float() * (qj <= qmax)[..., None]
    win = torch.cat([qoh, torch.stack([flag, eff_j.float() / dc, needed_j, same, up, down, (fj == 1).float()], -1)], -1)
    win = win * wvalid[..., None]
    return static, win, wvalid


class Obs:
    __slots__ = ("static", "win", "wmask", "mask", "player")

    def __init__(self, static, win, wmask, mask, player):
        self.static, self.win, self.wmask, self.mask, self.player = static, win, wmask, mask, player


class VecEnv:
    """N independent games. Player 0 opens round 1; a game is over when a player hits 0 dice."""

    def __init__(self, spec: Spec, n: int):
        self.spec, self.n = spec, n
        D, H = spec.dice, spec.max_hist
        self.dice = torch.zeros(n, 2, D, dtype=torch.long)
        self.nd = torch.full((n, 2), D, dtype=torch.long)
        z = lambda *s: torch.zeros(*s, dtype=torch.long)
        self.cur, self.opener, self.bq, self.bf, self.hlen = z(n), z(n), z(n), z(n), z(n)
        self.hq, self.hf = z(n, H), z(n, H)

    def _roll(self, idx):
        D = self.spec.dice
        vals = torch.randint(1, 7, (idx.numel(), 2, D))
        alive = torch.arange(D)[None, None, :] < self.nd[idx][:, :, None]
        self.dice[idx] = vals * alive

    def reset(self):
        idx = torch.arange(self.n)
        self.nd[:] = self.spec.dice
        self.cur[:] = 0
        self.opener[:] = 0
        self.bq[:] = 0
        self.bf[:] = 0
        self.hlen[:] = 0
        self._roll(idx)

    def observe(self, idx) -> Obs:
        p = self.cur[idx]
        hand = self.dice[idx, p]
        counts = F.one_hot(hand, 7).sum(1)
        cq, cf = self.bq[idx], self.bf[idx]
        mask = legal_mask(self.spec, cq, cf)
        static, win, wmask = make_features(
            self.spec, counts, self.nd[idx, p], self.nd[idx, 1 - p], cq, cf,
            self.hq[idx], self.hf[idx], self.hlen[idx], self.opener[idx], p, mask)
        return Obs(static, win, wmask, mask, p)

    def step(self, idx, action, shaping: float = 0.0):
        """Apply one action per game in idx. Returns (done [k] bool, winner [k], shape [k]).

        `shape` is the acting player's potential-based shaping reward: with potential
        phi_i = shaping * (opp_dice - own_dice) (having fewer dice is good in this variant) and phi = 0
        at game end, each challenge moves it by +/-shaping and the terminal step cancels the rest, so a
        player's shaping rewards sum to exactly 0 over a game (policy-invariant reward densification).
        """
        spec = self.spec
        k = idx.numel()
        done = torch.zeros(k, dtype=torch.bool)
        winner = torch.full((k,), -1, dtype=torch.long)
        shape = torch.zeros(k)
        is_liar = action == spec.n_raise

        r = ~is_liar
        if r.any():
            ri = idx[r]
            q, f = action[r] // 6 + 1, action[r] % 6 + 1
            pos = self.hlen[ri]
            self.hq[ri, pos], self.hf[ri, pos] = q, f
            self.hlen[ri] = pos + 1
            self.bq[ri], self.bf[ri] = q, f
            self.cur[ri] = 1 - self.cur[ri]

        if is_liar.any():
            pos_in_k = is_liar.nonzero(as_tuple=True)[0]
            li = idx[is_liar]
            caller = self.cur[li]
            fb, qb = self.bf[li], self.bq[li]
            d = self.dice[li]
            match = (d == fb[:, None, None]) | ((d == 1) & (fb != 1)[:, None, None])
            valid = match.sum((1, 2)) >= qb
            win_ch = torch.where(valid, 1 - caller, caller)
            lose_ch = 1 - win_ch
            nd_c, nd_o = self.nd[li, caller], self.nd[li, 1 - caller]
            self.nd[li, win_ch] -= 1
            over = self.nd[li, win_ch] == 0
            if shaping:
                step_gain = torch.where(win_ch == caller, shaping, -shaping)
                terminal_gain = -shaping * (nd_o - nd_c).float()
                shape[pos_in_k] = torch.where(over, terminal_gain, step_gain)
            done[pos_in_k[over]] = True
            winner[pos_in_k[over]] = win_ch[over]
            cont = ~over
            if cont.any():
                ci = li[cont]
                self.opener[ci] = lose_ch[cont]
                self.cur[ci] = lose_ch[cont]
                self.bq[ci], self.bf[ci], self.hlen[ci] = 0, 0, 0
                self._roll(ci)
        return done, winner, shape


def py_features(state, spec: Spec):
    """Same inputs as VecEnv.observe, built from a Python GameState (bots, exploitability tools)."""
    from nfsp_model import NFSPConfig, encode_state_shared, legal_action_mask

    cfg = NFSPConfig(dice_count=spec.dice, arch="shared_face")
    mask = legal_action_mask(state, cfg)
    static, seq = encode_state_shared(state, cfg, mask)
    counts = torch.zeros(1, 7, dtype=torch.long)
    for d in state.my_dice:
        counts[0, d] += 1
    hist = state.bid_history
    hq = torch.tensor([[b.quantity for _, b in hist] or [0]])
    hf = torch.tensor([[b.face for _, b in hist] or [0]])
    opener = hist[0][0] if hist else state.my_index
    info = info_features(spec, counts, torch.tensor([len(state.my_dice)]),
                         torch.tensor([state.dice_counts[1 - state.my_index]]), hq, hf,
                         torch.tensor([len(hist)]), torch.tensor([opener]), torch.tensor([state.my_index]))
    static = torch.cat([static, info[0]])
    K, m = spec.window, min(spec.window, len(state.bid_history))
    win = torch.zeros(K, spec.bid_dim)
    wmask = torch.zeros(K, dtype=torch.bool)
    if m > 0:
        win[K - m:] = seq[-m:]
        wmask[K - m:] = True
    return static, win, wmask, mask
