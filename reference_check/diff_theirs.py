"""Side-by-side check, part 2 (JAX): DeepMind's own v_trace / get_loss_v / get_loss_nerd, assembled exactly
as in RNaDSolver.loss, on the batch exported by diff_ours.py."""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnad_ref as rnad

d = {k: jnp.asarray(v) for k, v in np.load(sys.argv[1]).items() if k != "row"}
ETA, BETA, CLIP = 0.2, 2.0, 1e4


def loss(logit, v_online):
    legal = d["legal"]
    pi = rnad._legal_policy(logit, legal)
    log_pi = rnad.legal_log_policy(logit, legal)
    log_policy_reg = log_pi - d["log_reg"]          # reference: log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)
    vts, hps, pts = [], [], []
    for player in range(2):
        vt, hp, pt = rnad.v_trace(d["v_target"], d["valid"], d["player_id"], d["acting_policy"], pi, log_policy_reg,
                                  rnad._player_others(d["player_id"], d["valid"], player), d["actions_oh"],
                                  d["rewards"][:, :, player], player, lambda_=1.0, c=1.0, rho=np.inf, eta=ETA)
        vts.append(vt); hps.append(hp); pts.append(pt)
    loss_v = rnad.get_loss_v([v_online] * 2, vts, hps)
    is_vector = jnp.expand_dims(jnp.ones_like(d["valid"]), axis=-1)
    loss_nerd = rnad.get_loss_nerd([logit] * 2, [pi] * 2, pts, d["valid"], d["player_id"], legal, [is_vector] * 2,
                                   clip=CLIP, threshold=BETA)
    return loss_v + loss_nerd, (vts, pts, pi)


(_, (vts, pts, pi)), (g_logit, g_v) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(d["logit"], d["v_online"])
pid = np.asarray(d["player_id"]).astype(int)
vt = np.where(pid == 0, np.asarray(vts[0])[..., 0], np.asarray(vts[1])[..., 0])
q = np.where((pid == 0)[..., None], np.asarray(pts[0]), np.asarray(pts[1]))
adv = q - (np.asarray(pi) * q).sum(-1, keepdims=True)
np.savez(sys.argv[2], vh=vt, adv=np.clip(adv, -CLIP, CLIP), grad_logit=np.asarray(g_logit), grad_v=np.asarray(g_v)[..., 0])
print("done")
