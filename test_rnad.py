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


def _dummy_state():
    from game import Bid, GameState
    return GameState([3], 0, 2, [1, 1], 2, Bid(1, 4), [(1, Bid(1, 4))], 1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")
