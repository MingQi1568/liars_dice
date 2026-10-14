import os
import sys
import numpy as np
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
rep("grad wrt logits, legal actions", o["grad_logit"][rows], t["grad_logit"][valid], mask)
rep("grad wrt logits, illegal actions", o["grad_logit"][rows], t["grad_logit"][valid], ~mask)
