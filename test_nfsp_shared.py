"""Checks for the shared_face architecture. Run: python3 test_nfsp_shared.py"""
import os
import random

import torch

from game import Bid, Bot, Game, GameState, PlayerState
from bots import RandomBot, SmartBot
from nfsp_buffers import ReplayBuffer, ReservoirBuffer
from nfsp_model import (
    NFSPBot, NFSPConfig, S_FACE1, S_FACES, S_LIAR, STATIC_DIM, build_net, encode_state,
    forward_states, legal_action_mask,
)
import train_nfsp

CFG = NFSPConfig(dice_count=5, arch="shared_face")
QMAX, QCAP = CFG.max_quantity, CFG.max_quantity + 1
DEV = torch.device("cpu")


def mkstate(dice, cb=None, hist=(), my_index=0, dice_counts=(5, 5)):
    return GameState(my_dice=list(dice), my_index=my_index, num_players=2, dice_counts=list(dice_counts),
                     total_dice=sum(dice_counts), current_bid=cb, bid_history=list(hist), last_bidder=None)


def run_net(net, state):
    mask = legal_action_mask(state, CFG)
    feats = encode_state(state, CFG, mask)
    return forward_states(net, [feats], mask.unsqueeze(0), DEV)[0]


class Recorder(Bot):
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


def collect_states(n=300):
    sink = []
    random.seed(0)
    while len(sink) < n:
        players = [PlayerState("A", bot=Recorder(RandomBot(), sink)), PlayerState("B", bot=Recorder(SmartBot(), sink))]
        Game(players, starting_dice=5).play()
    return sink


def test_shapes_and_layout():
    torch.manual_seed(0)
    net = build_net(CFG)
    states = collect_states()
    masks = torch.stack([legal_action_mask(s, CFG) for s in states])
    feats = [encode_state(s, CFG, m) for s, m in zip(states, masks)]
    assert all(f[0].shape == (STATIC_DIM,) for f in feats)
    assert all(f[1].shape[1] == CFG.bid_feature_dim for f in feats)
    out = forward_states(net, feats, masks, DEV)
    assert out.shape == (len(states), CFG.n_actions), out.shape
    assert torch.isfinite(out).all()


def test_no_nan_when_all_masked():
    net = build_net(CFG)
    state = mkstate([1, 2, 3, 4, 5], cb=Bid(2, 3))
    feats = encode_state(state, CFG)
    for mask in (torch.zeros(1, CFG.n_actions, dtype=torch.bool),        # terminal dummy
                 legal_action_mask(mkstate([1, 2, 3, 4, 5], cb=Bid(10, 6)), CFG).unsqueeze(0)):  # only liar legal
        out = forward_states(net, [feats], mask, DEV)
        assert torch.isfinite(out).all()
    only_liar = legal_action_mask(mkstate([1, 2, 3, 4, 5], cb=Bid(10, 6)), CFG)
    assert only_liar[:-1].sum() == 0 and only_liar[-1]


def test_out_of_range_claim():
    state = mkstate([2, 2, 3, 4, 5], cb=Bid(15, 3), hist=[(1, Bid(15, 3))])
    static, seq = encode_state(state, CFG)
    assert static[S_LIAR][6] == 1.0            # claim_impossible
    assert torch.isfinite(static).all() and torch.isfinite(seq).all()
    assert seq[0, :QMAX].sum() == 0            # quantity one-hot stays all-zero out of range
    assert static.max() <= 2.0 and static.min() >= -2.0


def test_no_bid_face_equivariance():
    torch.manual_seed(1)
    net = build_net(CFG)
    a = run_net(net, mkstate([1, 1, 4, 4, 4]))
    b = run_net(net, mkstate([1, 1, 3, 3, 3]))
    for q in range(1, QMAX + 1):
        base = (q - 1) * 6
        assert torch.allclose(a[base + 3], b[base + 2], atol=1e-6)   # 3 of a kind on 4 vs on 3
        assert torch.allclose(a[base + 2], b[base + 3], atol=1e-6)
    # random hands and random permutations of faces 2..6
    rng = random.Random(3)
    for _ in range(20):
        hand = [rng.randint(1, 6) for _ in range(5)]
        perm = dict(zip(range(2, 7), rng.sample(range(2, 7), 5)))
        perm[1] = 1
        o = run_net(net, mkstate(hand))
        p = run_net(net, mkstate([perm[d] for d in hand]))
        for q in range(QMAX):
            for f in range(1, 7):
                assert torch.allclose(o[q * 6 + f - 1], p[q * 6 + perm[f] - 1], atol=1e-6)
        assert torch.allclose(o[-1], p[-1], atol=1e-6)


def test_history_invariance():
    a = mkstate([1, 1, 4, 4, 4], cb=Bid(3, 4), hist=[(1, Bid(2, 4)), (0, Bid(3, 4))])
    b = mkstate([1, 1, 3, 3, 3], cb=Bid(3, 3), hist=[(1, Bid(2, 3)), (0, Bid(3, 3))])
    assert torch.equal(encode_state(a, CFG)[1], encode_state(b, CFG)[1])
    c = mkstate([1, 1, 4, 4, 4], cb=Bid(3, 4), hist=[(1, Bid(2, 3)), (0, Bid(3, 4))])
    assert not torch.equal(encode_state(a, CFG)[1], encode_state(c, CFG)[1])  # a real difference still shows


def test_min_next_bid_matches_bid_rules():
    for s in collect_states(150):
        static, _ = encode_state(s, CFG)
        for f in range(1, 7):
            expected = next((q for q in range(1, QMAX + 1) if s.current_bid is None or Bid(q, f) > s.current_bid), QCAP)
            got = static[S_FACE1][6] if f == 1 else static[S_FACES][(f - 2) * 10 + 8]
            assert round(float(got) * QCAP) == expected, (s.current_bid, f, expected, float(got) * QCAP)


def test_gradients_and_detach():
    torch.manual_seed(2)
    net = build_net(CFG)
    state = mkstate([1, 2, 3, 3, 6], cb=Bid(3, 4), hist=[(1, Bid(2, 3)), (0, Bid(2, 5)), (1, Bid(3, 4))])
    mask = legal_action_mask(state, CFG)
    feats = encode_state(state, CFG, mask)
    forward_states(net, [feats], mask.unsqueeze(0), DEV).sum().backward()
    assert all(p.grad is not None for p in net.parameters())

    net.zero_grad()
    forward_states(net, [feats], mask.unsqueeze(0), DEV)[0, -1].backward()   # liar output only
    for module in (net.q_score_mlp, net.face1_score_mlp):   # max-legal-raise feature is detached
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())
    assert net.shared_face_mlp[0].weight.grad.abs().sum() > 0   # but e_claim still trains the shared embedding


def test_old_checkpoint_still_loads():
    path = "checkpoints/flat_ep300000_final.pt"
    if not os.path.exists(path):
        print("  (skipped: no flat checkpoint present)")
        return
    bot = NFSPBot.from_checkpoint(path)
    assert bot.cfg.arch == "flat"
    assert bot.act(mkstate([1, 2, 3, 4, 5], cb=Bid(2, 3))) is not None


def test_training_steps_both_archs():
    for arch in ("flat", "shared_face"):
        cfg = NFSPConfig(dice_count=5, arch=arch)
        q, qt, sl = (build_net(cfg) for _ in range(3))
        qt.load_state_dict(q.state_dict())
        rl, res = ReplayBuffer(5000), ReservoirBuffer(5000)
        for _ in range(6):
            trans, sls = train_nfsp.run_episode(q, sl, cfg, eta=0.5, epsilon=0.1, device=DEV)
            for t in trans:
                rl.push(t)
            for s in sls:
                res.push(s)
        opt_q, opt_s = torch.optim.Adam(q.parameters(), 1e-3), torch.optim.Adam(sl.parameters(), 1e-3)
        lq = train_nfsp.train_q_step(q, qt, opt_q, rl.sample(32), 1.0, DEV, cfg)
        ls = train_nfsp.train_sl_step(sl, opt_s, res.sample(min(32, len(res))), DEV)
        assert torch.isfinite(torch.tensor([lq, ls])).all(), (arch, lq, ls)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")
