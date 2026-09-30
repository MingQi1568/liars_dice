"""Differential / statistical tests for vec_rollout.py's vectorized NFSP self-play.

Run: python3 test_vec_rollout.py
"""
import random

import torch

from bots import RandomBot, SmartBot
from game import Bot, Game, PlayerState
from nfsp_model import NFSPConfig, build_net, encode_bid_history_relative, _my_face_counts
from nfsp_buffers import ReplayBuffer, ReservoirBuffer
from train_nfsp import run_episode
from vec_env import Spec
from vec_rollout import play_batch, window_to_seq


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


def test_window_to_seq_matches_full_history_within_window():
    """Whenever a round's bid history fits in the window (<=8 bids), the vectorized
    reconstruction must exactly match nfsp_model's full-history python encoding -- the
    truncation this test is checking for only matters for rounds longer than that."""
    dice = 5
    spec, cfg = Spec(dice=dice), NFSPConfig(dice_count=dice, arch="shared_face")
    states = collect_states(600, dice=dice, seed=1)

    n, H = len(states), spec.max_hist
    hq, hf = torch.zeros(n, H, dtype=torch.long), torch.zeros(n, H, dtype=torch.long)
    hlen, opener, p = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    for i, s in enumerate(states):
        hlen[i] = len(s.bid_history)
        p[i] = s.my_index
        opener[i] = s.bid_history[0][0] if s.bid_history else s.my_index
        for j, (pl, b) in enumerate(s.bid_history):
            hq[i, j], hf[i, j] = b.quantity, b.face

    from vec_env import make_features
    counts = torch.zeros(n, 7, dtype=torch.long)
    hand = torch.zeros(n, dtype=torch.long)
    opp = torch.zeros(n, dtype=torch.long)
    cq, cf = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    for i, s in enumerate(states):
        for d in s.my_dice:
            counts[i, d] += 1
        hand[i] = len(s.my_dice)
        opp[i] = s.dice_counts[1 - s.my_index]
        cq[i] = s.current_bid.quantity if s.current_bid else 0
        cf[i] = s.current_bid.face if s.current_bid else 0
    from nfsp_model import legal_action_mask
    masks = torch.stack([legal_action_mask(s, cfg) for s in states])
    static, win, wmask = make_features(spec, counts, hand, opp, cq, cf, hq, hf, hlen, opener, p, masks)
    seq, lengths = window_to_seq(win, wmask, spec)

    checked_within, checked_over = 0, 0
    for i, s in enumerate(states):
        counts_i = _my_face_counts(s.my_dice)
        ref = encode_bid_history_relative(s, cfg, counts_i)
        if len(s.bid_history) <= spec.window:
            assert int(lengths[i]) == max(1, len(s.bid_history))
            got = seq[i, : int(lengths[i])]
            assert torch.allclose(got, ref, atol=1e-6), (i, len(s.bid_history))
            checked_within += 1
        else:
            checked_over += 1
    assert checked_within > 0
    print(f"  (within-window: {checked_within}, over-window (approximated): {checked_over} "
          f"of {len(states)} sampled decision points)")


def test_play_batch_legal_and_zero_sum():
    dice = 2  # short games -> fast test, still exercises multi-round play
    spec, cfg = Spec(dice=dice), NFSPConfig(dice_count=dice, arch="shared_face")
    q_net, sl_net = build_net(cfg), build_net(cfg)
    torch.manual_seed(0)
    transitions, sl_samples = play_batch(q_net, sl_net, cfg, spec, eta=0.3, epsilon=0.2, n_games=64)
    assert len(transitions) > 0
    rewards = [t.reward for t in transitions if t.done]
    assert len(rewards) == 128  # 64 games x 2 players, each contributes exactly one terminal transition
    assert sum(rewards) == 0  # each game's winner (+1) and loser (-1) cancel exactly
    assert all(r in (1.0, -1.0) for r in rewards)
    assert len(sl_samples) <= len(transitions)


def test_play_batch_matches_run_episode_statistics():
    """Same (randomly-initialized) nets and NFSP hyperparameters through both the sequential
    run_episode loop and the vectorized play_batch: different RNG paths, but average episode
    length (transitions per game, both players) should agree -- the same kind of statistical
    check test_vec_env.py already uses for the raw engine."""
    dice = 3
    spec, cfg = Spec(dice=dice), NFSPConfig(dice_count=dice, arch="shared_face")
    torch.manual_seed(7)
    q_net, sl_net = build_net(cfg), build_net(cfg)
    eta, epsilon = 0.2, 0.1

    random.seed(7)
    n_py = 400
    py_lens = []
    for _ in range(n_py):
        transitions, _ = run_episode(q_net, sl_net, cfg, eta, epsilon, torch.device("cpu"))
        py_lens.append(len(transitions))
    py_len = sum(py_lens) / len(py_lens)

    torch.manual_seed(9)
    n_vec = 1500
    transitions, _ = play_batch(q_net, sl_net, cfg, spec, eta, epsilon, n_games=n_vec)
    vec_len = len(transitions) / n_vec

    print(f"  (avg transitions/game (both players): sequential {py_len:.1f} vs vectorized {vec_len:.1f})")
    assert abs(py_len - vec_len) / py_len < 0.15, (py_len, vec_len)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")
