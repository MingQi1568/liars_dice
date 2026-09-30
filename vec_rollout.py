"""Vectorized NFSP self-play: many games advance in lockstep as tensor ops (vec_env.py's
engine) instead of one Python game at a time, to remove the per-decision Python overhead
that dominates train_nfsp.py's run_episode (encode_state_shared's per-item feature loops,
and _single_forward's single-item collate). The network architecture and checkpoint format
are unchanged -- only the rollout is vectorized -- so this is a drop-in producer of the same
(transitions, sl_samples) shape run_episode returns, for a batch of games instead of one.

Known approximation: a round's bid history is truncated to the last `spec.window` (8) bids,
same tradeoff already made and validated for R-NaD's NFSPReg (rnad.py) -- reproduces the
sequential SharedFaceNet's action probabilities to 1e-6 there because rounds this long are
rare in practice. See test_vec_rollout.py for how often it actually bites and an exact-match
check for rounds that stay within the window.
"""
import torch

from nfsp_buffers import Transition
from nfsp_model import NFSPConfig, forward_batch
from vec_env import Spec, VecEnv

NEG = float("-inf")


def window_to_seq(win, wmask, spec: Spec):
    """(win[n,K,D], wmask[n,K]) -> (seq[n,K,D], lengths[n]) for SharedFaceNet's
    pack_padded_sequence. Same left-pad -> right-pad reindexing as rnad.py's NFSPReg,
    which this mirrors (validated there to reproduce the sequential net's probabilities)."""
    B, K, D = win.shape
    m = wmask.sum(1)
    j = torch.arange(K, device=win.device)[None, :]
    src = (j + (K - m)[:, None]).clamp(max=K - 1)
    seq = win.gather(1, src[..., None].expand(-1, -1, D)) * (j < m[:, None])[..., None]
    return seq, m.clamp(min=1)


@torch.no_grad()
def play_batch(q_net, sl_net, cfg: NFSPConfig, spec: Spec, eta: float, epsilon: float, n_games: int):
    """Play n_games full self-play games at once. Mirrors train_nfsp.run_episode's semantics
    exactly (anticipatory mode sampled per player per episode, epsilon-greedy only in
    best-response mode, SL reservoir samples only from the greedy best-response branch) but
    batched. Returns (transitions, sl_samples) in run_episode's own format, for all n_games
    combined -- a drop-in replacement for `n_games` calls to run_episode."""
    env = VecEnv(spec, n_games)
    env.reset()
    mode_br = torch.rand(n_games, 2) < eta          # mode_br[g,p]: True = best-response, False = avg
    traj = [([], []) for _ in range(n_games)]        # traj[g][p] = list of (feats, action, mask)
    sl_samples = []
    winner = torch.full((n_games,), -1, dtype=torch.long)

    active = torch.arange(n_games)
    while active.numel():
        obs = env.observe(active)
        seq, lengths = window_to_seq(obs.win, obs.wmask, spec)

        q_logits = forward_batch(q_net, (obs.static, seq, lengths), obs.mask).masked_fill(~obs.mask, NEG)
        sl_logits = forward_batch(sl_net, (obs.static, seq, lengths), obs.mask).masked_fill(~obs.mask, NEG)

        is_br = mode_br[active, obs.player]
        rand_pick = is_br & (torch.rand(active.numel()) < epsilon)
        greedy_pick = is_br & ~rand_pick

        avg_action = torch.multinomial(torch.softmax(sl_logits, dim=-1), 1).squeeze(1)
        greedy_action = q_logits.argmax(dim=1)
        rand_action = torch.multinomial(obs.mask.float(), 1).squeeze(1)
        action = torch.where(greedy_pick, greedy_action, torch.where(rand_pick, rand_action, avg_action))

        active_l, player_l = active.tolist(), obs.player.tolist()
        len_l, action_l, greedy_l = lengths.tolist(), action.tolist(), greedy_pick.tolist()
        for k in range(len(active_l)):
            g, p = active_l[k], player_l[k]
            feats = (obs.static[k], seq[k, :len_l[k]])
            entry = (feats, action_l[k], obs.mask[k])
            traj[g][p].append(entry)
            if greedy_l[k]:
                sl_samples.append(entry)

        done, w, _ = env.step(active, action)
        winner[active[done]] = w[done]
        active = active[~done]

    transitions = []
    winner_l = winner.tolist()
    for g in range(n_games):
        for p in range(2):
            t = traj[g][p]
            n = len(t)
            reward = 1.0 if p == winner_l[g] else -1.0
            for i in range(n):
                feats, act_idx, mask = t[i]
                done_i = i == n - 1
                r = reward if done_i else 0.0
                next_feats = t[i + 1][0] if not done_i else None
                next_mask = t[i + 1][2] if not done_i else None
                transitions.append(Transition(feats, act_idx, r, next_feats, done_i, mask, next_mask))
    return transitions, sl_samples
