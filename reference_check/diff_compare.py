import os
import sys
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
SC = os.path.dirname(os.path.abspath(__file__))
b = np.load(f"{SC}/diff_batch.npz"); o = np.load(f"{SC}/diff_ours_out.npz"); t = np.load(f"{SC}/diff_theirs_out.npz")
valid = b["valid"] > 0
rows = b["row"][valid]
mask = o["mask"][rows]
def rep(name, ours, theirs, m=None):
    diff = np.abs(ours - theirs) if m is None else np.abs(ours - theirs)[m]
    scale = np.abs(theirs if m is None else theirs[m]).max()
    print(f"{name:34s} max |ours - theirs| = {diff.max():.3e}   (largest value {scale:.3e})")
rep("value targets (v-trace)", o["vh"][rows], t["vh"][valid])
rep("advantages, legal actions", o["adv"][rows], t["adv"][valid], mask)
rep("grad wrt online value", o["grad_v"][rows], t["grad_v"][valid])
rep("  ... ours x 2", 2 * o["grad_v"][rows], t["grad_v"][valid])
rep("grad wrt logits, legal actions", o["grad_logit"][rows], t["grad_logit"][valid], mask)
rep("grad wrt logits, illegal actions", o["grad_logit"][rows], t["grad_logit"][valid], ~mask)
# hypothesis: the only policy-loss difference is centering by the mean over ALL actions instead of legal ones
lg = torch.tensor(o["logits"][rows], requires_grad=True)
m = torch.tensor(mask)
adv = torch.tensor(o["adv"][rows])
A = m.shape[-1]
lc = lg - (lg * m).sum(-1, keepdim=True) / A
force = torch.where(adv > 0, lc < 2.0, lc > -2.0)
(torch.tensor(o["w"][rows]) * -(m * force * lc * adv).sum(-1)).sum().backward()
rep("  ... ours with all-action centering", lg.grad.numpy(), t["grad_logit"][valid], mask)
