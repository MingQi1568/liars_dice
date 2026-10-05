# Checking our R-NaD against DeepMind's reference implementation

`rnad_ref.py` is DeepMind's reference R-NaD (OpenSpiel commit 894710bb, Apache 2.0) with one keyword
rename for current JAX. `our_liars_dice.py` registers our 1-die rules (1s wild, 1 the lowest face) as an
OpenSpiel game, so the reference can run on exactly our game. See `RNAD_FINDINGS.md`, "Reference
comparison", for what these found.

## Setup (separate venv; the project itself does not need JAX)

    python3.12 -m venv osenv
    ./osenv/bin/python -m pip install open_spiel jax jaxlib dm-haiku optax chex

## Run the reference R-NaD

    ./osenv/bin/python ref_run.py DELTA_M TOTAL_STEPS EVAL_EVERY SEED OUT_TAG [GAME]
    # e.g. our game, delta_m 10k:
    ./osenv/bin/python ref_run.py 10000 100000 5000 2 O2_ours_dm10k our_liars_dice_d1

Logs exact NashConv (raw and with the reference's threshold/discretization) of the target network.
GAME defaults to OpenSpiel's `liars_dice` (6s wild, the highest face).

## Side-by-side loss check

Feeds one batch from our engine through DeepMind's `v_trace` / `get_loss_v` / `get_loss_nerd` and through
our training helpers, in float64, and compares value targets, advantages and gradients:

    python3 diff_ours.py diff_batch.npz diff_ours_out.npz                  # project python (torch)
    ./osenv/bin/python diff_theirs.py diff_batch.npz diff_theirs_out.npz   # reference (JAX)
    python3 diff_compare.py

Expected with our default options: value targets and advantages agree to ~1e-15; the value gradient is
half the reference's (our 0.5 factor; `--value-coef 2` matches); the logit gradient differs until NeuRD
centering is over all actions (`--center all`). `os_bridge.py` maps policies between our `exact_d1` arrays
and OpenSpiel tabular policies.
