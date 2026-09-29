"""Differential tests: the vectorized engine/features must match the original Python engine.

Run: python3 test_vec_env.py
"""
import random

import torch

from bots import RandomBot, SmartBot
from game import Bid, Bot, Game, GameState, PlayerState
from nfsp_model import NFSPConfig, legal_action_mask
from vec_env import Spec, VecEnv, legal_mask, make_features, py_features

import torch.nn.functional as F


class Recorder(Bot):
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


def collect_states(n, dice=5, seed=0):
    sink = []
    random.seed(seed)
    while len(sink) < n:
        ps = [PlayerState("A", bot=Recorder(RandomBot(), sink)), PlayerState("B", bot=Recorder(SmartBot(), sink))]
        Game(ps, starting_dice=dice).play()
    return sink[:n]


def arrays_from_states(states, spec):
    """Load Python GameStates into the tensors make_features expects."""
    n, H = len(states), spec.max_hist
    counts = torch.zeros(n, 7, dtype=torch.long)
    hq = torch.zeros(n, H, dtype=torch.long)
    hf = torch.zeros(n, H, dtype=torch.long)
    cols = {k: torch.zeros(n, dtype=torch.long) for k in ("hand", "opp", "cq", "cf", "hlen", "opener", "p")}
    for i, s in enumerate(states):
        for d in s.my_dice:
            counts[i, d] += 1
        cols["hand"][i] = len(s.my_dice)
        cols["opp"][i] = s.dice_counts[1 - s.my_index]
        cols["cq"][i] = s.current_bid.quantity if s.current_bid else 0
        cols["cf"][i] = s.current_bid.face if s.current_bid else 0
        cols["hlen"][i] = len(s.bid_history)
        cols["p"][i] = s.my_index
        cols["opener"][i] = s.bid_history[0][0] if s.bid_history else s.my_index
        for j, (pl, b) in enumerate(s.bid_history):
            hq[i, j], hf[i, j] = b.quantity, b.face
    return counts, hq, hf, cols


def test_legal_mask_matches_engine():
    for dice in (1, 2, 5):
        spec, cfg = Spec(dice=dice), NFSPConfig(dice_count=dice)
        cqs, cfs, refs = [], [], []
        for q in range(0, spec.qmax + 2):
            for f in range(0, 7):
                if (q == 0) != (f == 0):
                    continue
                cqs.append(q)
                cfs.append(f)
                st = GameState([1], 0, 2, [1, 1], 2, Bid(q, f) if q else None, [], None)
                refs.append(legal_action_mask(st, cfg))
        got = legal_mask(spec, torch.tensor(cqs), torch.tensor(cfs))
        assert torch.equal(got, torch.stack(refs)), dice


def test_features_match_python_encoder():
    for dice in (5, 2, 1):
        spec, cfg = Spec(dice=dice), NFSPConfig(dice_count=dice, arch="shared_face")
        states = collect_states(400, dice=dice, seed=dice)
        counts, hq, hf, c = arrays_from_states(states, spec)
        masks = torch.stack([legal_action_mask(s, cfg) for s in states])
        static, win, wmask = make_features(spec, counts, c["hand"], c["opp"], c["cq"], c["cf"], hq, hf,
                                           c["hlen"], c["opener"], c["p"], masks)
        for i, s in enumerate(states):
            ps, pw, pm, pmask = py_features(s, spec)
            assert torch.equal(pmask, masks[i])
            assert torch.allclose(ps, static[i], atol=1e-6), (dice, i, (ps - static[i]).abs().max())
            assert torch.equal(pm, wmask[i])
            assert torch.allclose(pw, win[i], atol=1e-6), (dice, i)


def _scripted_round(dice_p, bids, caller_is_next):
    """Play one round in the original engine with fixed dice and a fixed bid sequence, then a liar call."""
    script = list(bids) + ["liar"]

    class Scripted(Bot):
        def act(self, state):
            return script.pop(0)

    ps = [PlayerState("A", dice=list(dice_p[0]), bot=Scripted()), PlayerState("B", dice=list(dice_p[1]), bot=Scripted())]
    g = Game(ps, starting_dice=len(dice_p[0]))
    for p, d in zip(ps, dice_p):
        p.dice = list(d)
    orig_roll = PlayerState.roll
    PlayerState.roll = lambda self: None
    try:
        g.play_round()
    finally:
        PlayerState.roll = orig_roll
    winner_idx = next((i for i, p in enumerate(ps) if p.num_dice == 0), -1)
    return [ps[0].num_dice, ps[1].num_dice], g.current_player, winner_idx


def test_round_resolution_matches_engine():
    rng = random.Random(5)
    for dice in (1, 3, 5):
        spec = Spec(dice=dice)
        for _ in range(300):
            hands = [[rng.randint(1, 6) for _ in range(dice)] for _ in range(2)]
            legal_bids, cur = [], None
            for _ in range(rng.randint(1, 6)):
                opts = [Bid(q, f) for q in range(1, spec.qmax + 1) for f in range(1, 7) if cur is None or Bid(q, f) > cur]
                if not opts:
                    break
                cur = rng.choice(opts)
                legal_bids.append(cur)
            if not legal_bids:
                continue
            ref_nd, ref_next, ref_win = _scripted_round(hands, legal_bids, None)

            env = VecEnv(spec, 1)
            env.reset()
            env.dice[0] = torch.tensor([h + [0] * (dice - len(h)) for h in hands])
            idx = torch.tensor([0])
            for b in legal_bids:
                env.step(idx, torch.tensor([(b.quantity - 1) * 6 + (b.face - 1)]))
            done, winner, _ = env.step(idx, torch.tensor([spec.n_raise]))
            assert env.nd[0].tolist() == ref_nd, (hands, legal_bids, env.nd[0].tolist(), ref_nd)
            if ref_win >= 0:
                assert bool(done[0]) and int(winner[0]) == ref_win
            else:
                assert not bool(done[0]) and int(env.cur[0]) == ref_next and int(env.opener[0]) == ref_next
                assert int(env.hlen[0]) == 0 and int(env.bf[0]) == 0


def test_random_play_statistics_match_engine():
    """Uniform-random legal play in both engines: game length and first-mover win rate agree."""
    dice = 3
    spec = Spec(dice=dice)
    random.seed(11)

    class UniformLegal(Bot):
        def act(self, state):
            mask = legal_action_mask(state, NFSPConfig(dice_count=dice))
            i = random.choice(mask.nonzero(as_tuple=True)[0].tolist())
            from nfsp_model import action_to_bid
            return action_to_bid(i, NFSPConfig(dice_count=dice))

    steps, wins0 = 0, 0
    n_py = 1500
    for _ in range(n_py):
        cnt = []
        ps = [PlayerState(n, bot=Recorder(UniformLegal(), cnt)) for n in "AB"]
        w = Game(ps, starting_dice=dice).play()
        steps += len(cnt)
        wins0 += w == 0
    py_len, py_w0 = steps / n_py, wins0 / n_py

    torch.manual_seed(3)
    n = 6000
    env = VecEnv(spec, n)
    env.reset()
    active, total, w0 = torch.arange(n), 0, 0
    while active.numel():
        obs = env.observe(active)
        probs = obs.mask.float()
        a = torch.multinomial(probs / probs.sum(1, keepdim=True), 1).squeeze(1)
        done, winner, _ = env.step(active, a)
        total += active.numel()
        w0 += int((winner[done] == 0).sum())
        active = active[~done]
    vec_len, vec_w0 = total / n, w0 / n
    assert abs(py_len - vec_len) / py_len < 0.05, (py_len, vec_len)
    assert abs(py_w0 - vec_w0) < 0.04, (py_w0, vec_w0)
    print(f"  (avg decisions/game: python {py_len:.1f} vs vec {vec_len:.1f}; seat-0 win rate {py_w0:.3f} vs {vec_w0:.3f})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")
