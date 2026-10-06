"""Side-by-side check, part 1 (torch): build one batch with our engine, compute OUR value targets,
advantages and loss gradients with the real training helpers, and export the batch in the reference's
time-major [T, B] layout for diff_theirs.py."""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
from rnad import NEG, collect, neurd_loss, two_player_vtrace, vtrace_advantage
from rnad_net import InfoSetNet
from vec_env import Spec, VecEnv

ETA, BETA, CLIP, ALPHA, SCALE = 0.2, 2.0, 1e4, 0.4, 25.0
spec = Spec(dice=1)
torch.manual_seed(11)


def scaled_net(seed):
    torch.manual_seed(seed)
    n = InfoSetNet(spec, 32).double()
    with torch.no_grad():
        n.pi_head.weight.mul_(SCALE)
        n.pi_head.bias.mul_(SCALE)
    return n


net, reg_cur, reg_prev, vnet, vonline = (scaled_net(s) for s in (1, 2, 3, 4, 5))
torch.manual_seed(12)
traj = collect(VecEnv(spec, 96), net)
S, A = traj.S, spec.n_actions
st = traj.static.double()
mask = traj.mask
with torch.no_grad():
    lp = lambda n: torch.log_softmax(n.policy_logits(st, traj.win, traj.wmask, mask).masked_fill(~mask, NEG), -1).masked_fill(~mask, 0.0)
    logreg = ALPHA * lp(reg_cur) + (1 - ALPHA) * lp(reg_prev)
    v = vnet.value(st, traj.win, traj.wmask)                       # target-network value
logits = net.policy_logits(st, traj.win, traj.wmask, mask).detach().requires_grad_(True)
v_on = vonline.value(st, traj.win, traj.wmask).detach().requires_grad_(True)
logp = torch.log_softmax(logits.masked_fill(~mask, NEG), -1)
pi, logpi = logp.exp().detach(), logp.masked_fill(~mask, 0.0).detach()
logmu = traj.logmu.double()
print(f"rows {S}, max |behaviour logprob - pi logprob| = {(logmu - logpi.gather(1, traj.act[:, None]).squeeze(1)).abs().max():.2e}")

pen_row = ETA * (pi * (logpi - logreg)).sum(-1)                  # expected-KL penalty (reference mode)
vh, gm = two_player_vtrace(traj, v, logmu, logreg.gather(1, traj.act[:, None]).squeeze(1), ETA, None, 1.0, 1.0, pen_row)
adv = vtrace_advantage(pi, ETA * (logpi - logreg), traj.act, logmu, gm).clamp(-CLIP, CLIP)
n_p = torch.bincount(traj.player, minlength=2).double()
w = 1.0 / n_p[traj.player]
(w * neurd_loss(logits, mask, adv, BETA)).sum().backward()
(w * (v_on - vh) ** 2).sum().backward()

# ---- export in the reference's time-major layout
order = torch.argsort(traj.env, stable=True)
rows_of = [order[traj.env[order] == e].numpy() for e in range(traj.n_envs)]
T, B = max(len(r) for r in rows_of) + 1, traj.n_envs
z = lambda *shape: np.zeros(shape)
out = dict(valid=z(T, B), player_id=z(T, B), actions_oh=z(T, B, A), acting_policy=z(T, B, A) + 1.0 / A,
           logit=z(T, B, A), legal=np.ones((T, B, A)), log_reg=z(T, B, A), v_target=z(T, B, 1), v_online=z(T, B, 1),
           rewards=z(T, B, 2), row=-np.ones((T, B), dtype=np.int64))
win = traj.winner.numpy()
for b, rows in enumerate(rows_of):
    for t, r in enumerate(rows):
        out["valid"][t, b] = 1
        out["player_id"][t, b] = int(traj.player[r])
        out["actions_oh"][t, b, int(traj.act[r])] = 1
        out["acting_policy"][t, b] = pi[r].numpy()
        out["logit"][t, b] = logits[r].detach().numpy()
        out["legal"][t, b] = mask[r].numpy()
        out["log_reg"][t, b] = logreg[r].numpy()
        out["v_target"][t, b, 0] = v[r]
        out["v_online"][t, b, 0] = v_on[r].detach()
        out["row"][t, b] = r
    out["rewards"][len(rows) - 1, b] = [1.0, -1.0] if win[b] == 0 else [-1.0, 1.0]
ours = dict(vh=vh.numpy(), adv=adv.numpy(), grad_logit=logits.grad.numpy(), grad_v=v_on.grad.numpy(),
            mask=mask.numpy(), logits=logits.detach().numpy(), player=traj.player.numpy(), w=w.numpy(),
            # the flat batch, so make_golden.py can store a fixture the regular test suite rebuilds without JAX
            env=traj.env.numpy(), act=traj.act.numpy(), logmu=logmu.numpy(), final=traj.final.numpy(),
            shape=traj.shape.double().numpy(), offsets=np.array(traj.offsets), winner=traj.winner.numpy(),
            v=v.numpy(), logreg=logreg.numpy(), v_online=v_on.detach().numpy())
np.savez(sys.argv[1], **out)
np.savez(sys.argv[2], **ours)
print(f"exported T={T} B={B}; {int((adv.abs() > 0).sum())} nonzero advantages; "
      f"{int(((logits.detach() - (logits.detach() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)).abs() > BETA).sum())} centered logits beyond beta")
