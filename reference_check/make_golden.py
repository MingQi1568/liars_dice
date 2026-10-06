"""Store the side-by-side batch plus DeepMind's outputs (mapped to our flat rows) as a small fixture,
test_data/reference_losses_d1.npz, so test_rnad.py checks our losses against the reference without JAX.
Run after diff_ours.py and diff_theirs.py (see README)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
b = np.load(os.path.join(HERE, "diff_batch.npz"))
o = np.load(os.path.join(HERE, "diff_ours_out.npz"))
t = np.load(os.path.join(HERE, "diff_theirs_out.npz"))
valid = b["valid"] > 0
rows = b["row"][valid]
S = o["act"].shape[0]
ref = {}
for k in ("vh", "adv", "grad_logit", "grad_v"):
    arr = t[k][valid]
    out = np.zeros((S,) + arr.shape[1:])
    out[rows] = arr
    ref["ref_" + k] = out
keep = ("env", "player", "act", "logmu", "final", "shape", "offsets", "winner", "mask", "logits", "v", "logreg",
        "v_online")
path = os.path.join(os.path.dirname(HERE), "test_data", "reference_losses_d1.npz")
os.makedirs(os.path.dirname(path), exist_ok=True)
np.savez_compressed(path, **{k: o[k] for k in keep}, **ref)
print(f"wrote {path} ({os.path.getsize(path)} bytes, {S} rows)")
