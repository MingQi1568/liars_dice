# Liar's dice: NFSP and R-NaD agents

**Start with `NEXT_STEPS_5DICE.md`** (current plan: R-NaD at 5 dice), then `RNAD_FINDINGS.md` (full record of the
R-NaD work and why the settings are what they are).

## Layout

- `game.py`, `bots.py`, `best_bot.py`: Python game engine and heuristic bots (SmartBot, BestBot).
- `vec_env.py`: vectorized torch engine (~60x faster; differential-tested against `game.py`).
- `rnad.py`, `rnad_net.py`, `train_rnad.py`: R-NaD trainer, matched numerically to DeepMind's reference.
- `nfsp_model.py` and the NFSP scripts: the earlier NFSP agent (baseline / reference policy).
- `exact_d1.py`, `diag_d1.py`, `exploit_d1.py`, `cfr_d1.py`, `init_cfr_d1.py`: exact 1-die tools (NashConv,
  CFR+, diagnostics).
- `reference_check/`: DeepMind's `rnad.py`, our game as an OpenSpiel game, side-by-side loss harness.
- `sweeps/`: launch and plot scripts for every experiment in `RNAD_FINDINGS.md`.

## Conventions

- Tests: `python3 test_rnad.py && python3 test_vec_env.py` (plain scripts, no pytest). The golden tests pin our
  losses to DeepMind's; if one fails, a change altered the algorithm.
- Training on CPU with `--threads 1`; one run per machine where possible (memory bandwidth, not cores, limits
  concurrent runs).
- Judge R-NaD settings by exact NashConv at 1 die (raw and fine-tuned), and at 5 dice by an exploitability
  proxy, not by win rate against bots. Compare configurations by where they plateau.
- `checkpoints/` is gitignored. Commit or push only when asked.
