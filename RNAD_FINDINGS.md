# R-NaD self-play for Liar's Dice — status and findings

Branch: `rnad-vectorized`. This is a from-scratch, CPU-optimized vectorized self-play
engine plus a DeepNash-style R-NaD (Regularized Nash Dynamics) trainer for 1v1 Liar's
Dice, built as a candidate replacement/improvement over the NFSP bot already in this
repo (`nfsp_model.py`, `train_nfsp.py`). **It has not yet been shown to beat NFSP.**
This doc is a handoff of what's been built and what we've learned, for whoever
(human or Claude instance) picks this up next.

Reference: the DeepNash / R-NaD paper (`Mastering Stratego with model-free multiagent
reinforcement learning`-family work; local copy was read from `deepnash.pdf`, not
included in this repo).

## Files added on this branch

- `vec_env.py` — vectorized engine: N games advance in lockstep as tensor ops
  (~60x faster than the original `game.py` per-game loop). `Spec(dice, window)`,
  `VecEnv.step/observe`, `legal_mask`, `make_features` (matches
  `nfsp_model.encode_state_shared`'s static features + a fixed window of the last
  8 bids), and `py_features` so the same feature pipeline works from a Python
  `GameState` for bots/exploitability tools.
- `rnad_net.py` — `RNaDNet`: same face-shared-weight architecture idea as NFSP's
  `shared_face` net, plus a GRU over the last-K-bids window, with a value head added
  (R-NaD needs `policy_logits()`, `value()`, and an optional per-action `q` head).
- `rnad.py` — `Config`, `RNaD` trainer (`learn_step`, `two_player_vtrace`, `distill`,
  checkpointing), and three regularization-policy classes: `UniformReg` (paper
  default start), `NetReg` (wraps a frozen copy of any net — used for the periodic
  outer-iteration reference reset), `NFSPReg` (wraps a trained NFSP `shared_face`
  SL-net as the reference — our addition, not in the paper).
- `train_rnad.py` — CLI trainer. Evaluates exact NashConv at `--dice 1`
  (`exploit_d1.py`), or win-rate vs `SmartBot`/`BestBot`/an NFSP checkpoint at
  `--dice 5`. Flags of note: `--init-reg-nfsp` (seed the reference from an NFSP
  checkpoint), `--distill-steps` (supervised KL warm-start onto that reference before
  R-NaD starts), `--shaping` (potential-based reward shaping, off by default),
  `--iter-steps` (Δm — steps before the reference resets to a target-net copy).
- `exploit_d1.py` — exact best-response / NashConv solver for the 1-die game
  (4,096 histories × 6 dice, fully enumerated).
- `cfr_d1.py` — CFR+ solver on the same 1-die game, as a near-equilibrium reference
  (NashConv ≈ 0.009 after 400 iterations).
- `test_vec_env.py`, `test_rnad.py` — 11 tests total, all passing: differential tests
  of the vectorized engine/features against the original Python engine, a
  brute-force check of the v-trace math, a Monte-Carlo cross-check of the exact
  solver, and an `NFSPReg` fidelity check.

Run tests with `python3 test_vec_env.py && python3 test_rnad.py`.

## Reward function (as currently implemented)

Three layers, stacked:

1. **Base game reward** — terminal `+1`/`-1` for winning/losing the whole game
   (first to 0 dice wins; challenge winner loses a die — see `game.py`), plus an
   **optional** potential-based shaping term (`--shaping`, default 0, off by default
   to match the paper). Shaping uses `φ = shaping · (opp_dice − own_dice)`, `φ = 0`
   at game end; it fires only on a "liar" (challenge) action, crediting the caller
   `+shaping` if the call was correct and `-shaping` if not, with a terminal
   correction so the total nets to exactly 0 per player per game (`vec_env.py`
   `VecEnv.step`, verified by `test_shaping_sums_to_zero_per_player`). This is
   provably policy-invariant (Ng/Harada/Russell 1999) — it densifies the learning
   signal without changing what's optimal, *as long as* the terminal-correction term
   is present. (We also worked through why a naive "+1/dice_max per round" reward
   without that correction is **not** policy-invariant — it rewards margin-of-victory,
   not just winning, because the per-game total isn't path-independent.)
2. **R-NaD's regularization transform**, applied on top of (1), not instead of it:
   `pen = eta * (logpi_online(a|s) - logpi_reg(a|s))`; acting player's reward becomes
   `r_env - pen`, the other player's becomes `-r_env + pen`. This penalizes straying
   from the reference policy and is what makes the fixed point of the whole procedure
   a Nash equilibrium rather than just "whatever beats the current opponent" (unlike,
   e.g., naive self-play PPO — see discussion below).
3. This transformed reward feeds two-player v-trace (`two_player_vtrace`, no
   bootstrapping, computed backward over whole games) to build the value target and
   the NeuRD policy-gradient advantage.

## Why not just use PPO / plain self-play policy gradient?

Discussed at length this session; summary: PPO's clipped-trust-region update assumes
an approximately stationary environment, but in self-play the "environment" is the
opponent, which is also being updated — this can cycle around the equilibrium
(Rock-Paper-Scissors dynamics) rather than converge to it, and PPO has no mechanism
analogous to R-NaD's reference-policy regularization or NFSP's time-averaging to damp
that. Hidden information (opponent's dice) also makes PPO's per-state value baseline
very high-variance. R-NaD's reward transform is specifically constructed (proven in
the paper) so its fixed point is a Nash equilibrium of the real game.

## What we tried, in order, and what happened

### 1. From scratch (uniform reference), 5 dice
0% win rate vs SmartBot/BestBot after 600 steps, with or without shaping. Far too
slow to reach anything useful in an 8-hour budget.

### 2. NFSP-seeded reference + supervised distillation warm start
Added `NFSPReg` (wraps the trained NFSP `shared_face` SL-net,
`checkpoints/shared_face/nfsp_latest.pt`, verified to reproduce its action
probabilities to 1e-6) as the initial regularization policy, plus `RNaD.distill()`:
a short supervised KL-minimization warm start (~200-300 steps, ~3 min) that pulls the
online net's policy onto the reference before R-NaD proper begins. This unblocks
training — runs now start around 45-62% win rate vs SmartBot/BestBot/NFSP instead of
0%, matching (this being a supervised copy) the NFSP baseline.

### 3. Moving-reference runs (the paper's standard setup: periodic outer-iteration reset)
Several runs varying eta (0.2-0.5), lr (1e-5 - 3e-4), `iter_steps`/Δm (300-1000),
all starting from the NFSP-distilled policy. **Consistent result: performance
degrades monotonically from the distilled starting point.** E.g. the longest run
(η=0.5, lr=3e-5, iter_steps=300, shaping=0.1, ~1.2h/2318 steps,
`checkpoints/rnad_d5_run1/`): 62.3%/49.8%/47.2% (SmartBot/BestBot/NFSP) at the
distilled start → 33.0%/35.2%/42.0% by step 2000 → still declining at the time cutoff.
Every checkpoint saved along the way was worse than the one before it. No config
tried beat the distilled starting point.

### 4. Fixed-reference convergence tests (isolate inner-loop behavior)
Set `iter_steps` far above the step budget so the reference is pinned for the whole
run — no outer-loop resets — to test whether the R-NaD inner loop (v-trace + NeuRD)
mechanically converges at all, separate from any moving-target instability.

- **1 die, exact NashConv available** (lr=1e-4, β=6, batch=4096, uniform reference,
  `checkpoints/rnad_d1_fixedref/`): NashConv fell smoothly and monotonically,
  1.41 → 0.54 over 2600 steps (~30 min), clearly flattening (diminishing returns:
  Δ0.73 in the first 800 steps, Δ0.03 in the last 800). **The inner loop does
  converge** — but plateaus far short of equilibrium (CFR+ reaches 0.009 in 400
  iterations on the same game). Conclusion: the learning-step math is sound; the
  bottleneck at this scale is sampling noise from hidden information (each game's
  outcome is highly stochastic in the opponent's unseen dice), not an implementation
  bug or instability.
- **5 dice, lr=1e-5** (η=0.5, β=6, shaping=0.1, NFSP-distilled start,
  `checkpoints/rnad_d5_fixedref/`, ~1h/1498 steps): gradient norm fell cleanly
  (474→37) but `kl` (mean KL(online‖reference) per decision) kept climbing
  (0.31→0.50) without flattening; win rate rose *above* the distilled baseline
  briefly around step 300-450 (peak 64%/55%/54%) then declined back toward/below it
  by step 1450 (52%/44%/46%).
- **5 dice, lr=5e-5** (same else, `checkpoints/rnad_d5_fixedref_5e5/`,
  ~2h/2806 steps): **worse than lr=1e-5.** `kl` climbed faster and *accelerated* in
  the second half (0.37→0.84 over 2800 steps; rate roughly doubled from the first
  half to the second). `ent` rose to a peak near step 800 (1.49) then steadily
  collapsed to 1.26 by step 2800 — the policy became more confident/narrow while
  simultaneously drifting further from the reference, a bad combination. Win rate vs
  BestBot declined steadily the entire run (51% → 30%). Gradient norm alone looked
  flat/converged the whole time (~30-35 band from step 400 on) — **a misleading
  signal in isolation**, since it's the combined gradient of the policy loss *and*
  the value loss (which converges faster), and `kl`/`ent` kept moving underneath a
  flat `gn`. Takeaway: **raising lr did not speed up convergence — it broke it.**

## What we use to judge "convergence" at 5 dice (no exact NashConv available there)

Track four scalars together each log step, don't trust any single one alone:
- `kl` — mean KL(online policy ‖ reference policy) per decision. Should flatten if
  the online net is settling into the regularized fixed point.
- `ent` — policy entropy (`-Σ π log π`, nats). Should flatten, not keep rising or
  suddenly collapse.
- `gn` — gradient norm. Should decay and flatten, but is **contaminated by the value
  loss** and can look converged well before the policy actually has (see the lr=5e-5
  run above) — never rely on it alone.
- `val` — value-loss magnitude. Should stabilize.
- As a secondary, noisier sanity check (not the real objective — R-NaD optimizes
  toward equilibrium, not "beat this specific bot"): win rate vs SmartBot/BestBot/NFSP.

## Working hypotheses / where this stands

- R-NaD's core learning step (v-trace + NeuRD regularized update) is implemented
  correctly and does converge given a truly fixed reference and enough steps — shown
  cleanly at 1 die.
- At 5 dice, inner-loop convergence has **not** been observed within any budget
  tried so far (≤2800 steps, ≤2h). Lower lr (1e-5) is closer to well-behaved than
  higher (5e-5) — suggests the self-play sampling noise (hidden dice → high
  per-game outcome variance) requires either a much smaller lr, many more steps,
  and/or a larger batch (more games/step) than anything tried so far, rather than
  faster learning being achievable by raising lr.
- **We have not demonstrated R-NaD beating the existing NFSP bot on the full game.**
  Every 5-dice run plateaus at or below the NFSP-distilled starting point within the
  budgets tried.
- The originally-planned 8-hour production run has **not** been launched — evidence
  so far doesn't support that it would beat NFSP, and better to resolve the
  convergence question at smaller scale first.

## Hyperparameters reference

| Param | `Config` default | DeepNash paper | What we've tried at 5 dice |
|---|---|---|---|
| `games_per_step` (batch) | 1024 | 768 | 1024, 2048 |
| `iter_steps` (Δm) | 1000 | 10k→100k→35k schedule over 7.21M steps | 300, 1000, 1e6 (pinned, for fixed-ref tests) |
| `eta` (η) | 0.2 | 0.2 | 0.2, 0.5 |
| `lr` | 3e-4 | 5e-5 | 1e-5, 3e-5, 5e-5, 1e-4, 3e-4 |
| `gamma` (target EMA) | 0.01 | 0.001 | 0.01 |
| `beta` (β, NeuRD clip) | 2.0 | 2 | 6 (found better than 2 in 1-die tuning) |
| `shaping` (ours, not in paper) | 0.0 | n/a | 0.0, 0.1, 0.3 |

Optimizer: Adam, betas=(0, 0.999), eps=1e-8 (matches paper). Games are full episodes,
no discounting/bootstrapping (γ=1 implicitly; finite episodic game).

## NFSP baseline (used as the R-NaD reference/distillation target)

`checkpoints/shared_face/nfsp_latest.pt` — `shared_face` architecture (32,243 params
per net; face-shared weight-tied MLP over faces 2-6 + separate small heads for face-1
and "liar", GRU(64) over bid history, quantity embeddings). Trained 1.2M episodes
(~8.6h, CPU) with anticipatory η=0.1, ε 0.08→0.01, RL buffer 100k / SL buffer 500k,
batch 128, RL lr 1e-4→1e-5, SL lr 1e-3→1e-4, all with linear decay over 80% of
training. Final eval: ~54%/50% vs SmartBot/BestBot. See `project-nfsp-training-facts`
and `project-rnad-findings` memory files (if available in your Claude memory) for
more detail; also `train_nfsp_shared_face.log` in the repo root for the full run log.

## Suggested next steps (not yet tried)

1. Larger `games_per_step` (more games per step, not necessarily more steps) to
   directly cut sampling noise, as an alternative/complement to lowering lr further.
2. lr below 1e-5 at 5 dice, given 5e-5 clearly regressed relative to 1e-5.
3. An adaptive `iter_steps`: advance the outer iteration when `kl`/`ent`/`gn` all
   show a flat trailing-window slope, instead of a fixed guess — discussed, not
   implemented.
4. Once (if) a config shows real fixed-reference convergence at 5 dice, re-test the
   moving-reference (paper-standard) setup with that config before considering an
   8-hour production run.

## Checkpoints produced this session (gitignored, not in this PR)

`checkpoints/rnad_d5_run1/`, `checkpoints/rnad_d1_fixedref/`,
`checkpoints/rnad_d5_fixedref/`, `checkpoints/rnad_d5_fixedref_5e5/` — all local only
(`checkpoints/` is gitignored). Share these separately if the next machine needs them
instead of retraining; otherwise the distillation warm-start makes re-establishing a
comparable starting point cheap (~3 min) as long as
`checkpoints/shared_face/nfsp_latest.pt` is available.
