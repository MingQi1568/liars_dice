"""Vectorized exact tools for the 1-die game (a single round): policy values, counterfactual Q-values,
best responses, NashConv, and an exact, noise-free tabular R-NaD used as an "oracle".

A policy is an array pi[H, 6, A]: for history index h (see exploit_d1.enumerate_histories), the die of
the player to act there (0..5 = faces 1..6), and action a (raises 0..R-1, then liar = R). Histories of
equal length share the acting player (length % 2), so everything is processed one depth level at a
time with the two players' dice carried as a (d0, d1) 6x6 axis. exploit_d1.Game1 is the slow
reference implementation these functions are tested against.
"""
import numpy as np

from exploit_d1 import enumerate_histories
from game import Bid


class Tree:
    def __init__(self, qmax: int = 2):
        hists = enumerate_histories(qmax)
        idx = {h: i for i, h in enumerate(hists)}
        H, R = len(hists), qmax * 6
        A = R + 1
        depth = np.array([len(h) for h in hists])
        child = -np.ones((H, R), dtype=np.int64)
        for i, h in enumerate(hists):
            cur = Bid(*h[-1]) if h else None
            for a in range(R):
                q, f = a // 6 + 1, a % 6 + 1
                if cur is None or Bid(q, f) > cur:
                    child[i, a] = idx[h + ((q, f),)]
        legal = np.zeros((H, A), dtype=bool)
        legal[:, :R] = child >= 0
        legal[:, R] = depth > 0
        pay = np.zeros((H, 6, 6))                       # payoff to player 0 if the actor at h calls liar
        d = np.arange(1, 7)
        for i, h in enumerate(hists):
            if h:
                q, f = h[-1]
                m = ((d == f) | ((d == 1) & (f != 1))).astype(int)
                valid = (m[:, None] + m[None, :]) >= q
                actor = len(h) % 2
                pay[i] = np.where(np.where(valid, 1 - actor, actor) == 0, 1.0, -1.0)
        self.hists, self.idx, self.H, self.R, self.A = hists, idx, H, R, A
        self.child, self.legal, self.pay, self.depth = child, legal, pay, depth
        self.levels = [np.nonzero(depth == L)[0] for L in range(depth.max() + 1)]

    # ---------------------------------------------------------------- helpers
    def uniform(self) -> np.ndarray:
        p = np.broadcast_to(self.legal[:, None, :], (self.H, 6, self.A)).astype(float)
        return p / p.sum(-1, keepdims=True)

    def softmax(self, y: np.ndarray) -> np.ndarray:
        m = self.legal[:, None, :]
        z = np.where(m, y, -np.inf)
        z = z - z.max(-1, keepdims=True)
        e = np.where(m, np.exp(z), 0.0)
        return e / e.sum(-1, keepdims=True)

    def _child_vals(self, V, hs):
        """[n, A, 6, 6] values (player-0 terms) of every action at histories hs: raises -> child V, liar -> payoff."""
        ch = self.child[hs]
        vc = np.where((ch >= 0)[:, :, None, None], V[np.maximum(ch, 0)], 0.0)
        return np.concatenate([vc, self.pay[hs][:, None]], axis=1)

    @staticmethod
    def _over_actor(x, actor):
        """Broadcast an [n, 6(actor die), A] array to [n, A, d0, d1] along the actor's die axis."""
        x = x.transpose(0, 2, 1)
        return x[:, :, :, None] if actor == 0 else x[:, :, None, :]

    # ---------------------------------------------------------------- core passes
    def reaches(self, pi: np.ndarray) -> np.ndarray:
        """Rr[p, h, d]: product of player p's own action probabilities along h, given p's die d."""
        Rr = np.ones((2, self.H, 6))
        for L, hs in enumerate(self.levels[:-1]):
            actor = L % 2
            ch = self.child[hs]
            ok = ch >= 0
            hh = np.repeat(hs[:, None], self.R, 1)[ok]
            aa = np.tile(np.arange(self.R), (len(hs), 1))[ok]
            cc = ch[ok]
            Rr[actor, cc] = Rr[actor, hh] * pi[hh, :, aa]
            Rr[1 - actor, cc] = Rr[1 - actor, hh]
        return Rr

    def evaluate(self, pi, eta=0.0, logratio=None, Rr=None):
        """Exact values and counterfactual Q-values of `pi` played by both players.

        With eta > 0 the rewards are R-NaD-transformed: the acting player pays eta*log(pi/pi_reg)
        for its action and the other player receives it (logratio[h, d, a] = log pi - log pi_reg,
        in the actor's die axis). Returns (V0, Q):
          V0[h, d0, d1]: expected payoff to player 0 from h onward (both dice known);
          Q[h, d, a]:    counterfactual value of action a for the actor at (h, die d), in the
                         actor's own terms, conditioned on reaching h (opponent reach normalized).
        """
        if Rr is None:
            Rr = self.reaches(pi)
        V = np.zeros((self.H, 6, 6))
        Q = np.zeros((self.H, 6, self.A))
        for L in reversed(range(len(self.levels))):
            hs = self.levels[L]
            actor = L % 2
            sgn = 1.0 if actor == 0 else -1.0                         # actor's terms -> player-0 terms
            tot = self._child_vals(V, hs)                             # [n, A, 6, 6]
            if eta:
                tot = tot - sgn * eta * self._over_actor(logratio[hs], actor)
            V[hs] = (self._over_actor(pi[hs], actor) * tot).sum(1)
            ro = Rr[1 - actor, hs]                                    # opponent reach [n, 6]
            if actor == 0:
                q = np.einsum("nade,ne->nda", tot, ro)
            else:
                q = np.einsum("nade,nd->nea", tot, ro)
            Q[hs] = sgn * q / np.maximum(ro.sum(-1), 1e-300)[:, None, None]
        return V, Q

    def best_response(self, pi: np.ndarray, P: int):
        """Value of a best-responding player P against pi, and the BR policy (deterministic, P's nodes only)."""
        Rr = self.reaches(pi)
        V = np.zeros((self.H, 6, 6))                                  # payoff to P
        br = np.zeros((self.H, 6, self.A))
        sP = 1.0 if P == 0 else -1.0
        for L in reversed(range(len(self.levels))):
            hs = self.levels[L]
            actor = L % 2
            tot = self._child_vals(V, hs)       # raise children already hold P's values ...
            tot[:, self.R] *= sP                # ... the liar payoff is in player-0 terms
            if actor == P:
                ro = Rr[1 - P, hs]
                if P == 0:
                    score = np.einsum("nade,ne->nda", tot, ro)
                else:
                    score = np.einsum("nade,nd->nea", tot, ro)
                score = np.where(self.legal[hs][:, None, :], score, -np.inf)
                a_star = score.argmax(-1)                               # [n, 6]
                br[hs[:, None], np.arange(6)[None, :], a_star] = 1.0
                V[hs] = (self._over_actor(br[hs], actor) * tot).sum(1)
            else:
                V[hs] = (self._over_actor(pi[hs], actor) * tot).sum(1)
        return float(V[0].mean()), br

    def nash_conv(self, pi: np.ndarray):
        b0, _ = self.best_response(pi, 0)
        b1, _ = self.best_response(pi, 1)
        return b0 + b1, (b0, b1)

    def seat0_value(self, pi: np.ndarray) -> float:
        V, _ = self.evaluate(pi)
        return float(V[0].mean())

    def visitation(self, pi: np.ndarray, Rr=None) -> np.ndarray:
        """vis[h, d]: probability that the actor at h reaches h holding die d, under self-play with pi."""
        if Rr is None:
            Rr = self.reaches(pi)
        vis = np.zeros((self.H, 6))
        for L, hs in enumerate(self.levels):
            actor = L % 2
            vis[hs] = Rr[actor, hs] / 6.0 * Rr[1 - actor, hs].mean(-1, keepdims=True)
        return vis

    def merge(self, pi_seat: np.ndarray, pi_other: np.ndarray, seat: int) -> np.ndarray:
        """Joint policy array where `seat` plays pi_seat and the other seat plays pi_other."""
        out = pi_other.copy()
        mine = (self.depth % 2) == seat
        out[mine] = pi_seat[mine]
        return out


# -------------------------------------------------------------------- post-processing (DeepNash fine-tuning)
def threshold_policy(pi, legal, thr):
    if thr <= 0:
        return pi
    keep = (pi >= thr) | (pi.max(-1, keepdims=True) < thr)
    p = np.where(keep & legal[:, None, :], pi, 0.0)
    return p / p.sum(-1, keepdims=True)


def discretize_policy(pi, n=32):
    """Round each distribution to multiples of 1/n, largest probabilities first (OpenSpiel's FineTuning)."""
    if n <= 0:
        return pi
    flat = pi.reshape(-1, pi.shape[-1])
    out = np.zeros_like(flat)
    roundup = np.ceil(flat * n).astype(int)
    order = np.argsort(-flat, axis=-1)
    left = np.full(flat.shape[0], n)
    rows = np.arange(flat.shape[0])
    for k in range(flat.shape[1]):
        a = order[:, k]
        x = np.minimum(roundup[rows, a], left)
        out[rows, a] += x
        left -= x
    out[rows, order[:, 0]] += left
    return (out / n).reshape(pi.shape)


# -------------------------------------------------------------------- exact (tabular, noise-free) R-NaD
def rnad_exact(tree: Tree, eta: float, outer: int, inner: int, lr: float = 1.0, tol: float = 0.0,
               pi0=None, log=None, weighting: str = "infoset"):
    """Exact R-NaD with tabular logits: each inner step is a full-width NeuRD/mirror-ascent update on
    exact counterfactual Q-values of the eta-transformed game; after `inner` steps (or when the policy
    moves less than `tol`) the current policy becomes the next regularization policy.

    weighting="infoset": every infoset takes the same-size step (the idealized dynamics).
    weighting="visit":   each infoset's step is scaled by its self-play visitation probability
                         (relative to the most-visited infoset at that point) - the expected update
                         of sampled on-policy R-NaD with a tabular policy, i.e. infinite batch.
    Returns a list of (outer_iter, inner_steps_used, NashConv, policy)."""
    pi = tree.uniform() if pi0 is None else pi0.copy()
    y = np.where(tree.legal[:, None, :], np.log(np.maximum(pi, 1e-300)), 0.0)
    pi_reg = pi.copy()
    out = []
    for m in range(outer):
        used = inner
        for n in range(inner):
            logratio = np.where(tree.legal[:, None, :],
                                np.log(np.maximum(pi, 1e-300)) - np.log(np.maximum(pi_reg, 1e-300)), 0.0)
            Rr = tree.reaches(pi)
            _, Q = tree.evaluate(pi, eta, logratio, Rr)
            adv = Q - (pi * Q).sum(-1, keepdims=True)
            if weighting == "visit":
                vis = tree.visitation(pi, Rr)
                adv = adv * (vis / vis.max())[:, :, None]
            y = y + lr * np.where(tree.legal[:, None, :], adv, 0.0)
            y = y - np.where(tree.legal[:, None, :], y, -np.inf).max(-1, keepdims=True)
            new = tree.softmax(y)
            moved = np.abs(new - pi).max()
            pi = new
            if tol and moved < tol:
                used = n + 1
                break
        nc, _ = tree.nash_conv(pi)
        out.append((m + 1, used, nc, pi.copy()))
        if log:
            log(f"  outer {m + 1}: inner steps {used}, NashConv {nc:.4f}")
        pi_reg = pi.copy()
    return out
