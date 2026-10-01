"""Correctness checks for the R-NaD implementation. Run: python3 test_rnad.py"""
import torch

from rnad import Config, RNaD, collect, policy_probs, two_player_vtrace
from rnad_net import RNaDNet
from vec_env import Spec, VecEnv, py_features


def test_vtrace_matches_bruteforce_returns():
    """On-policy (all ratios 1), v-trace targets equal the exact return-to-go of the transformed rewards."""
    torch.manual_seed(0)
    spec = Spec(dice=2)
    env = VecEnv(spec, 64)
    net = RNaDNet(spec).eval()
    traj = collect(env, net, shaping=0.1)
    S = traj.S
    v = torch.randn(S)
    logreg_taken = -torch.rand(S)
    eta = 0.3
    vh, gm = two_player_vtrace(traj, v, traj.logmu, logreg_taken, eta)

    pen = eta * (traj.logmu - logreg_taken)
    r_env = traj.shape.clone()
    fin = traj.final
    r_env[fin] += torch.where(traj.winner[traj.env[fin]] == traj.player[fin], 1.0, -1.0)
    # rows per env in time order (rows are time-major, so a stable sort by env gives per-game order)
    order = torch.argsort(traj.env, stable=True)
    for e in range(traj.n_envs):
        rows = order[traj.env[order] == e]
        for i in (0, 1):
            # transformed reward each row gives player i: acting -> r_env - pen; other -> -r_env + pen
            r_i = torch.stack([(r_env[r] - pen[r]) if traj.player[r] == i else (-r_env[r] + pen[r]) for r in rows])
            future = torch.flip(torch.cumsum(torch.flip(r_i, [0]), 0), [0])          # sum_{t' >= t}
            after = torch.cat([future[1:], torch.zeros(1)])                             # sum_{t' > t}
            for k, r in enumerate(rows):
                if traj.player[r] == i:
                    assert abs(float(vh[r]) - float(future[k])) < 1e-4, (e, k)
                    assert abs(float(gm[r] + v[r]) - float(r_env[r] + after[k])) < 1e-4, (e, k)


def test_shaping_sums_to_zero_per_player():
    """Potential-based shaping with phi=0 at game end must telescope to exactly 0 for each player."""
    torch.manual_seed(4)
    spec = Spec(dice=4)
    traj = collect(VecEnv(spec, 300), RNaDNet(spec).eval(), shaping=0.2)
    assert traj.shape.abs().sum() > 0
    for i in (0, 1):
        own = torch.where(traj.player == i, traj.shape, -traj.shape)         # zero-sum: other player gets the negative
        tot = torch.zeros(traj.n_envs).index_add_(0, traj.env, own)
        assert tot.abs().max() < 1e-5, tot.abs().max()


def test_rollout_is_well_formed():
    spec = Spec(dice=3)
    env = VecEnv(spec, 200)
    traj = collect(env, RNaDNet(spec).eval())
    assert (traj.winner >= 0).all()
    assert int(traj.final.sum()) == 200
    assert traj.mask.gather(1, traj.act[:, None]).all()          # every sampled action was legal
    assert (traj.logmu <= 0).all()
    assert traj.offsets[-1] == traj.S


def test_learn_step_runs_and_changes_policy():
    torch.manual_seed(1)
    cfg = Config(dice=1, games_per_step=256, iter_steps=5, chunk=512)
    agent = RNaD(cfg)
    before = [p.clone() for p in agent.net.parameters()]
    for _ in range(6):                                            # crosses one outer-iteration boundary
        m = agent.learn_step()
        assert all(torch.isfinite(torch.tensor(float(m[k]))) for k in ("kl", "ent", "val", "pol", "gn"))
    assert agent.iter == 1 and agent.n_in_iter == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.net.parameters()))
    # checkpoint round trip
    other = RNaD(cfg)
    other.load_state_dict(agent.state_dict())
    static, win, wmask, mask = py_features(_dummy_state(), cfg.spec)
    a = policy_probs(agent.target, static[None], win[None], wmask[None], mask[None])
    b = policy_probs(other.target, static[None], win[None], wmask[None], mask[None])
    assert torch.allclose(a, b)


def test_critic_mode_runs():
    torch.manual_seed(2)
    agent = RNaD(Config(dice=1, games_per_step=256, iter_steps=4, chunk=512, adv_mode="critic"))
    for _ in range(5):
        m = agent.learn_step()
        assert all(torch.isfinite(torch.tensor(float(m[k]))) for k in ("kl", "ent", "val", "pol", "gn"))


def test_exact_solver_matches_simulation():
    """Exact expected payoff under a policy must match Monte-Carlo self-play (validates the solver,
    the simulator, and that both feature pipelines yield the same policy)."""
    from exploit_d1 import Game1, net_prob_fn, uniform_prob_fn
    spec = Spec(dice=1)
    g = Game1(uniform_prob_fn(spec), spec)
    nc, (br0, br1) = g.nash_conv()
    assert 1.4 < nc < 1.7 and br0 > 0.7 and br1 > 0.7       # uniform play is badly exploitable
    torch.manual_seed(7)
    net = RNaDNet(spec).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5 * torch.randn_like(p))
    g = Game1(net_prob_fn(net, spec), spec)
    exact = g.seat0_value()
    tr = collect(VecEnv(spec, 100000), net)
    mc = 2 * float((tr.winner == 0).float().mean()) - 1
    assert abs(exact - mc) < 0.012, (exact, mc)
    assert g.nash_conv()[0] > 0


def test_nfsp_reg_matches_nfsp_policy():
    """The NFSP-based regularization policy must reproduce the NFSP model's own action probabilities
    (histories up to the window length). Skipped when no checkpoint is present."""
    import os
    path = "checkpoints/shared_face/nfsp_latest.pt"
    if not os.path.exists(path):
        print("  (skipped: no NFSP checkpoint)")
        return
    from nfsp_model import NFSPBot, encode_state, forward_states, legal_action_mask
    from rnad import NFSPReg
    from test_vec_env import collect_states
    spec = Spec(dice=5)
    reg, bot = NFSPReg(path, spec), NFSPBot.from_checkpoint(path)
    states = [s for s in collect_states(300, dice=5, seed=3) if len(s.bid_history) <= spec.window]
    static, win, wmask, mask = (torch.stack(x) for x in zip(*[py_features(s, spec) for s in states]))
    got = reg.log_probs(static, win, wmask, mask).exp() * mask
    for i, s in enumerate(states):
        m = legal_action_mask(s, bot.cfg)
        with torch.no_grad():
            lg = forward_states(bot.sl_net, [encode_state(s, bot.cfg, m)], m.unsqueeze(0), "cpu")[0]
        ref = torch.softmax(lg.masked_fill(~m, float("-inf")), -1)
        assert (ref - got[i]).abs().max() < 1e-4


def test_entropy_schedule_matches_reference():
    """Same cases as the reference implementation's tests (open_spiel rnad_test.py / docstring)."""
    from rnad import EntropySchedule
    assert EntropySchedule([3, 5, 10], [2, 4, 1]).schedule == [0, 3, 6, 11, 16, 21, 26, 36]
    expected = [(0, False), (2 / 3, False), (1, True), (0, False), (2 / 3, False), (1, True), (0, False),
                (0.4, False), (0.8, False), (1, False), (1, True), (0, False), (1 / 3, False), (2 / 3, False),
                (1, False), (1, False), (1, True), (0, False), (1 / 3, False), (2 / 3, False), (1, False),
                (1, False), (1, True), (0, False)]
    sched = EntropySchedule([3, 5, 6], [2, 1, 1])
    for i, (a, u) in enumerate(expected):
        got_a, got_u = sched(i)
        assert abs(got_a - a) < 1e-9 and got_u == u, (i, got_a, got_u, a, u)


def test_vtrace_with_expected_penalty_matches_bruteforce():
    """With an explicit per-row penalty (the reference's expected KL), on-policy v-trace targets still
    equal the exact return-to-go of the transformed rewards for each player."""
    torch.manual_seed(3)
    spec = Spec(dice=2)
    traj = collect(VecEnv(spec, 48), RNaDNet(spec).eval(), shaping=0.1)
    v = torch.randn(traj.S)
    pen = 0.2 * torch.rand(traj.S)
    vh, gm = two_player_vtrace(traj, v, traj.logmu, traj.logmu, 0.2, pen=pen)
    r_env = traj.shape.clone()
    fin = traj.final
    r_env[fin] += torch.where(traj.winner[traj.env[fin]] == traj.player[fin], 1.0, -1.0)
    order = torch.argsort(traj.env, stable=True)
    for e in range(traj.n_envs):
        rows = order[traj.env[order] == e]
        for i in (0, 1):
            r_i = torch.stack([(r_env[r] - pen[r]) if traj.player[r] == i else (-r_env[r] + pen[r]) for r in rows])
            future = torch.flip(torch.cumsum(torch.flip(r_i, [0]), 0), [0])
            after = torch.cat([future[1:], torch.zeros(1)])
            for k, r in enumerate(rows):
                if traj.player[r] == i:
                    assert abs(float(vh[r]) - float(future[k])) < 1e-4
                    assert abs(float(gm[r] + v[r]) - float(r_env[r] + after[k])) < 1e-4


def test_reference_options_learn_and_round_trip():
    """Reference-mode training (infoset MLP, expected-KL reward, per-player loss, no grad clip, a
    multi-size schedule) runs, moves the regularization policy at the scheduled steps, and checkpoints."""
    from rnad import load_policy_net
    torch.manual_seed(5)
    cfg = Config(dice=1, games_per_step=256, chunk=512, net="infoset", hidden=32, reg_reward="expected",
                 loss_norm="per_player", grad_clip=0.0, sched_sizes=(3, 5), sched_repeats=(1, 1))
    agent = RNaD(cfg)
    iters = []
    for _ in range(13):
        m = agent.learn_step()
        assert all(torch.isfinite(torch.tensor(float(m[k]))) for k in ("kl", "ent", "val", "pol", "gn"))
        iters.append(agent.iter)
    assert iters[2] == 1 and iters[7] == 2 and iters[12] == 3 and iters[1] == 0 and iters[6] == 1, iters
    import os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "ck.pt")
    torch.save(agent.state_dict(), path)
    net, cfg2 = load_policy_net(path)
    static, win, wmask, mask = py_features(_dummy_state(), cfg.spec)
    a = policy_probs(agent.target, static[None], win[None], wmask[None], mask[None])
    b = policy_probs(net, static[None], win[None], wmask[None], mask[None])
    assert cfg2.net == "infoset" and torch.allclose(a, b)


def test_exact_tools_match_reference_solver():
    """The vectorized exact solver (exact_d1) agrees with the slow recursive one (exploit_d1.Game1)."""
    import numpy as np
    from exact_d1 import Tree, discretize_policy
    from exploit_d1 import Game1
    spec = Spec(dice=1)
    tree = Tree(spec.qmax)
    pi = tree.softmax(np.random.default_rng(1).normal(size=(tree.H, 6, tree.A)))
    g = Game1(lambda states: np.zeros((len(states), tree.A)), spec)
    g.probs = pi
    assert abs(tree.nash_conv(pi)[0] - g.nash_conv()[0]) < 1e-9
    assert abs(tree.seat0_value(pi) - g.seat0_value()) < 1e-9
    d = discretize_policy(pi, 32)
    assert np.allclose(d.sum(-1), 1) and np.allclose(d * 32, np.round(d * 32))


def test_infoset_loss_weights():
    """loss_norm='infoset': within each player, every distinct information set in the batch gets total
    policy-loss weight 1/(number of distinct infosets of that player)."""
    torch.manual_seed(6)
    agent = RNaD(Config(dice=1, games_per_step=300, chunk=512, net="infoset", hidden=16, loss_norm="infoset"))
    m = agent.learn_step()
    assert all(torch.isfinite(torch.tensor(float(m[k]))) for k in ("kl", "ent", "val", "pol", "gn"))
    from vec_env import STATIC_DIM
    traj = collect(VecEnv(Spec(dice=1), 300), agent.net)
    key = torch.cat([traj.player[:, None].float(), traj.static[:, STATIC_DIM:]], 1)
    _, inv, cnt = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    assert int(cnt.max()) > 1 and cnt.shape[0] < traj.S          # infosets genuinely repeat within a batch


def _dummy_state():
    from game import Bid, GameState
    return GameState([3], 0, 2, [1, 1], 2, Bid(1, 4), [(1, Bid(1, 4))], 1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")
