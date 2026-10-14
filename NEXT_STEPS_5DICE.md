# Handoff: training R-NaD on 5 dice (written 2026-10-14)

For whoever continues this on another machine (human or Claude). Read this first, then `RNAD_FINDINGS.md`
(the full record; its "Summary of findings" and "1-die tuning sweep" sections are the important ones).

## State of the code (branch `rnad-vectorized`, PR #3 into `main`)

- `rnad.py` / `train_rnad.py` match DeepMind's reference R-NaD (open_spiel `rnad.py`) numerically: losses and
  gradients agree to ~1e-14 on a fixed batch, with and without training-time fine-tuning
  (`test_matches_reference_losses*` in `test_rnad.py`). **The defaults are the reference settings.**
- **Every R-NaD result before commit a614b1a is invalid** (a NeuRD centering bug caused the "improves, then
  degrades after a few reference resets" pattern, including all earlier 5-dice runs). Don't compare against them.
- Run the tests after pulling: `python3 test_rnad.py && python3 test_vec_env.py` (22 checks, a few minutes).
- `checkpoints/` is gitignored: no trained models come with the repo. 1-die results are reproducible from
  `sweeps/*.sh`; `cfr_d1_policy.npy` (needed by `diag_d1.py` / `init_cfr_d1.py`) is regenerated with
  `python3 diag_d1.py cfr`.

## What we learned at 1 die (exact NashConv; raw [fine-tuned]; lower is better)

| Lever | Finding |
|---|---|
| Δm (steps between reference resets) | The most important. 50k beats 10k at every batch size; 10k plateaus ~0.245. Most progress comes in the first ~5-6 resets. Switching to 500k after the plateau gave only ~0.004 more. |
| Batch size | 4096 beats 512 per reset. 16384 (Δm 50k) pulls ahead of 4096 on raw NashConv from reset 4 (0.208 vs 0.214 at reset 6) and was still falling when stopped: **its plateau is still open**. Bigger batches cost proportionally more games. |
| Fine-tuning at test time (drop actions < 0.03, round to 1/32) | Large, free gain: ~0.21 -> ~0.10. Always evaluate and deploy the fine-tuned policy. |
| Fine-tuning during training (`--finetune-from`) | Made things worse. Off by default; leave it off. |
| NeuRD beta | Keep 2 (paper value). 4/8 halve raw NashConv briefly, then degrade (entropy collapse). |
| eta 0.5, target EMA 0.01 | No effect; keep 0.2 and 0.001. |
| Late lr decay (`--lr-points`) | Doesn't lower the plateau but makes the final checkpoint stable. |
| Starting from the CFR+ equilibrium | Self-play leaves it within 5k steps and settles on the same plateau, which is where sampled R-NaD settles at this scale. |

Best 1-die result: batch 4096 or 16384, Δm 50k -> ~0.21 raw / ~0.10-0.11 fine-tuned (DeepMind's own code on our
game reaches ~0.23 / 0.13). A perfect exploiter would win only ~52.5% against the fine-tuned policy. Against
BestBot at 1 die the fine-tuned bot wins 49.5% ± 0.7%, the same as the exact CFR+ equilibrium (49.9%), so
BestBot win rate cannot separate near-equilibrium policies.

The user's preference: **judge configurations by where they plateau, not by equal-compute comparisons**, and
don't call a run "no gain" before it has leveled off.

## 5 dice: what changes

- Hyperparameters don't transfer numerically. A 5-dice game is ~35 decisions (1 die: ~3) over several rounds,
  with far more information sets.
- Measured on an M-series Mac, one run alone, `--threads 1`: **~0.11 s/step at batch 512, ~0.86 s/step at batch
  4096**. So one Δm-50k iteration is ~1.5 h at batch 512 and ~12 h at batch 4096. Plan around this.
- **No exact NashConv at 5 dice.** The trainer's dice-5 eval reports win rate vs SmartBot / BestBot, which is a
  weak proxy (not exploitability; non-transitive). Use it only to catch collapsed runs.

## Recommended plan

1. **Build an exploitability proxy before tuning anything** (Path C in `RNAD_FINDINGS.md`): an approximate
   best response (train a DQN/PPO exploiter against the frozen policy, fixed budget, both seats, report its win
   margin) and/or local best response (belief over the opponent's dice from the policy's own probabilities, greedy
   response). **Validate it at 1 die first**: run it on 1-die checkpoints from the sweeps and check it ranks them
   like exact NashConv does. Include a known-bad policy (uniform / early checkpoint) to show the exploiter finds
   holes. The NFSP code's best-response learner can be reused as the DQN exploiter.
2. **Baseline 5-dice run** with the reference defaults and a DeepNash-style schedule (short resets early, long
   later):
   ```bash
   python3 train_rnad.py --dice 5 --games-per-step 512 --sched-sizes 10000 50000 --sched-repeats 10 1 \
     --total-steps 3000000 --eval-every 10000 --save-every 50000 --threads 1 --max-hours 24 \
     --checkpoint-dir checkpoints/rnad_d5_base
   ```
   Then compare batch sizes (512 / 2048 / 4096) and Δm by their plateaus under the proxy from step 1.
3. **Network:** the default `--net infoset` (MLP on the exact round encoding, 69 inputs at 5 dice) is the
   reference architecture and had the best 1-die floor (0.043). Also try the face-shared net with rank and context
   (`--net face --face-rank --info-ctx 64`; 1-die floor 0.11): weight sharing across faces may learn faster at
   5 dice, which 1 die can't show.
4. **Evaluate the fine-tuned policy** (`threshold_policy` + `discretize_policy` in `exact_d1.py`, or
   `post_process_policy` in `rnad.py`), not the raw one.

## Practical notes

- **Compute:** CPU is ~5x faster than Apple MPS here; use `--threads 1` (more threads didn't help). Runs that
  share a machine compete for memory bandwidth, not cores: on the Mac, 4 concurrent runs slowed batch-16384
  steps from 0.14 s to 0.6 s. Prefer one run per machine/instance. Training steps are sequential, so cloud
  machines speed up sweeps (many runs at once), not a single run. EC2 `c7a` spot is ~2¢ per core-hour.
- Long runs: `--max-hours` saves and stops cleanly; `--resume CKPT` continues exactly and accepts overrides for
  `--beta`, `--lr`, `--finetune-*` and `--sched-sizes/--sched-repeats` (schedules count from step 0).
  `--save-every N` keeps numbered checkpoints every N steps; `rnad_latest.pt` is written at every eval.
- On a laptop, keep it plugged in with the lid open; `caffeinate` can't prevent clamshell sleep on battery.
- `reference_check/` (DeepMind's code + the side-by-side harness) needs its own Python 3.12 venv with
  open_spiel/jax/dm-haiku/optax/chex; see its README. Only needed to re-verify against the reference.
- Plots: `sweeps/plot_long12h.py` is a template (palette, reset markers, by-step and by-games panels).
