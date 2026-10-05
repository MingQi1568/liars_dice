"""Run DeepMind's reference R-NaD (rnad_ref.py, one jnp.clip keyword rename) on OpenSpiel's liars_dice (1 die)
and log exact NashConv of the target network, raw and with the reference's own post-processing."""
import os
import sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability
import rnad_ref as rnad

GAME_NAME = sys.argv[6] if len(sys.argv) > 6 else "liars_dice"
if GAME_NAME == "our_liars_dice_d1":
    import our_liars_dice  # noqa: F401  (registers our rules as an OpenSpiel game)
GAME = pyspiel.load_game(GAME_NAME)
TAB = policy_lib.TabularPolicy(GAME)


def nash_conv_of(solver):
    env = solver._batch_of_states_as_env_step(TAB.states)
    out = {}
    for name, fn in (("raw", solver._network_jit_apply), ("finetuned", solver._network_jit_apply_and_post_process)):
        probs = np.asarray(jax.device_get(fn(solver.params_target, env)), dtype=np.float64)
        probs = probs * TAB.legal_actions_mask
        TAB.action_probability_array = probs / probs.sum(-1, keepdims=True)
        out[name] = exploitability.nash_conv(GAME, TAB)
    return out


def main():
    dm, total, every, seed, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    cfg = rnad.RNaDConfig(game_name=GAME_NAME, trajectory_max=14, batch_size=512, learning_rate=5e-5,
                          entropy_schedule_size=(dm,), entropy_schedule_repeats=(1,), seed=seed)
    s = rnad.RNaDSolver(cfg)
    print(f"config: {cfg}", flush=True)
    t0, hist = time.time(), []
    while s.learner_steps < total:
        s.step()
        if s.learner_steps % every == 0:
            nc = nash_conv_of(s)
            hist.append((s.learner_steps, nc["raw"], nc["finetuned"]))
            print(f"step {s.learner_steps} (iter {s.learner_steps // dm}): NashConv raw {nc['raw']:.4f} "
                  f"finetuned {nc['finetuned']:.4f} [{(time.time() - t0) / 60:.1f}m]", flush=True)
            json.dump(hist, open(f"{tag}.json", "w"))


if __name__ == "__main__":
    main()
