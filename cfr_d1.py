"""CFR+ on the 1-die-each game: a reference equilibrium to validate exploit_d1 and benchmark R-NaD.

Run: python3 cfr_d1.py [iterations]
"""
import sys

import numpy as np

from exploit_d1 import Game1
from vec_env import Spec


class CFR:
    def __init__(self, spec):
        # reuse Game1's tree (uniform policy is a placeholder; only the tree structure/payoffs are used)
        n_states = 4096 * 6
        A = spec.n_actions
        self.g = Game1(lambda states: np.full((len(states), A), 1.0 / A), spec)
        g = self.g
        self.A, self.n_raise = A, spec.n_raise
        H = len(g.hists)
        self.regret = np.zeros((H, 6, A))
        self.strat_sum = np.zeros((H, 6, A))
        self.legal = np.zeros((H, A), dtype=bool)
        for hi, h in enumerate(g.hists):
            for a, _ in g.children[hi]:
                self.legal[hi, a] = True
            if h:
                self.legal[hi, self.n_raise] = True
        self.t = 0

    def strategy(self, hi):
        r = np.maximum(self.regret[hi], 0) * self.legal[hi]
        tot = r.sum(-1, keepdims=True)
        uni = self.legal[hi] / self.legal[hi].sum()
        return np.where(tot > 0, r / np.maximum(tot, 1e-12), uni)

    def traverse(self, hi, reach):
        g = self.g
        h = g.hists[hi]
        actor = len(h) % 2
        opp = 1 - actor
        sigma = self.strategy(hi)                                    # [6, A]
        u_actor = np.zeros(6)
        u_opp = np.zeros(6)
        util = {}
        acts = [(a, ci) for a, ci in g.children[hi]]
        if h:
            acts = acts + [(self.n_raise, None)]
        for a, ci in acts:
            new_reach = [reach[0], reach[1]]
            new_reach[actor] = reach[actor] * sigma[:, a]
            if ci is None:                                            # liar call: terminal
                # payoff to actor for each actor die (vector over opp die) and to opp for each opp die
                ua = np.array([(new_reach[opp] * g.payoff(actor, dp, hi)).sum() for dp in range(1, 7)])
                uo = np.array([(new_reach[actor] * g.payoff(opp, do, hi)).sum() for do in range(1, 7)])
                ua_o = (ua, uo)
            else:
                ua_o = self._split(self.traverse(ci, new_reach), actor)
            util[a] = ua_o
            u_actor += sigma[:, a] * ua_o[0]
            u_opp += ua_o[1]
        for a, _ in acts:
            self.regret[hi, :, a] += util[a][0] - u_actor
        self.strat_sum[hi] += (reach[actor][:, None] * sigma) * self.t
        out = np.zeros((2, 6))
        out[actor], out[opp] = u_actor, u_opp
        return out

    @staticmethod
    def _split(out, actor):
        return out[actor], out[1 - actor]

    def iterate(self, n):
        for _ in range(n):
            self.t += 1
            self.traverse(0, [np.full(6, 1 / 6), np.full(6, 1 / 6)])
            np.maximum(self.regret, 0, out=self.regret)               # CFR+

    def average_policy(self):
        s = self.strat_sum * self.legal[:, None, :]
        tot = s.sum(-1, keepdims=True)
        uni = self.legal[:, None, :] / self.legal.sum(-1)[:, None, None]
        return np.where(tot > 0, s / np.maximum(tot, 1e-12), uni)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    spec = Spec(dice=1)
    cfr = CFR(spec)
    hidx = {h: i for i, h in enumerate(cfr.g.hists)}
    for _ in range(iters // 50):
        cfr.iterate(50)
        pol = cfr.average_policy()

        def prob_fn(states):
            out = np.zeros((len(states), spec.n_actions))
            for i, s in enumerate(states):
                h = tuple((b.quantity, b.face) for _, b in s.bid_history)
                out[i] = pol[hidx[h], s.my_dice[0] - 1]
            return out

        g = Game1(prob_fn, spec)
        nc, br = g.nash_conv()
        print(f"CFR+ iter {cfr.t}: NashConv={nc:.4f}  seat-0 value={g.seat0_value():+.4f}", flush=True)
