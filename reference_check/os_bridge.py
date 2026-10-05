"""Map between our exact_d1 policy arrays pi[H, 6, A] and OpenSpiel TabularPolicy on our_liars_dice_d1."""
import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
import our_liars_dice  # noqa: F401  (registers the game)

GAME = pyspiel.load_game("our_liars_dice_d1")


def histories():
    out = []
    def dfs(h):
        out.append(tuple(h))
        for a in range(12):
            if not h or a > h[-1]:
                dfs(h + [a])
    dfs([])
    return out


HISTS = histories()


def key(h, die):
    return f"p{len(h) % 2} d{die} b" + ",".join(str(b) for b in h)


def to_tabular(pi, tab=None):
    tab = tab or policy_lib.TabularPolicy(GAME)
    for hi, h in enumerate(HISTS):
        for d in range(1, 7):
            tab.action_probability_array[tab.state_lookup[key(h, d)]] = pi[hi, d - 1]
    return tab


def from_tabular(tab):
    pi = np.zeros((len(HISTS), 6, 13))
    for hi, h in enumerate(HISTS):
        for d in range(1, 7):
            pi[hi, d - 1] = tab.action_probability_array[tab.state_lookup[key(h, d)]]
    return pi
