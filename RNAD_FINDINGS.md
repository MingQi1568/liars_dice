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

> **Read "Reference comparison: the NeuRD centering bug" first.** Every run before it,
> including all 5-dice runs and the 20-hour production run, centered NeuRD logits over the
> legal actions instead of over all actions as DeepMind's code does. That one difference
> causes the degradation-after-reference-resets seen throughout this doc. Conclusions below
> that blame noise, compute scale or reset timing for it are superseded. Use
> `train_rnad.py --preset reference`, which now matches the reference implementation.

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
- `exact_d1.py` — vectorized exact tools for the 1-die game (policy values,
  counterfactual Q, best responses, NashConv, self-play visitation, DeepNash's
  threshold/discretize fine-tuning) and `rnad_exact`, a noise-free tabular R-NaD
  "oracle". Matches `exploit_d1.py` to 1e-9 and is ~6x faster.
- `diag_d1.py` — reproduces the Phase 1 diagnostics below (`cfr`, `oracle`, `floor`,
  `patch`, `finetune` subcommands).
- `rnad_net.py` also has `InfoSetNet`: the reference implementation's architecture
  (MLP 256x256 on an exact per-round information-set encoding, `vec_env.info_features`,
  stored after the first 76 feature columns).
- `train_rnad.py --preset reference` switches on DeepMind's reference settings, including
  the NeuRD centering fix (see "Reference comparison"); defaults are unchanged so old
  checkpoints resume exactly as before.
- `reference_check/` — DeepMind's reference R-NaD, our 1-die rules as an OpenSpiel game,
  and the side-by-side loss check (needs a separate JAX venv; see its README).
- `RNaDNet --face-rank --info-ctx N` — face-shared net with face-rank inputs and an MLP over
  the exact round encoding (see "Network floor").
- `test_vec_env.py`, `test_rnad.py` — 16 tests total, all passing: differential tests
  of the vectorized engine/features against the original Python engine, brute-force
  checks of the v-trace math (sampled and expected penalty), a Monte-Carlo cross-check
  of the exact solver, the exact tools vs the slow solver, the reference
  EntropySchedule test cases, reference-mode training + checkpointing, and an
  `NFSPReg` fidelity check.

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

**Correction (later research):** that argument holds for *naive* self-play PPO. PPO with
a KL pull toward a fixed or annealed "magnet" policy (MMD-style) is a recognised strong
baseline: Rudolph et al., *Reevaluating Policy Gradient Methods for
Imperfect-Information Games* (ICLR 2026, 7,000+ runs, exact exploitability) found
FP-, DO- and CFR-based deep methods "fail to outperform generic policy gradient
methods". Worth trying if R-NaD stays unstable.

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

*(Written before the Phase 1 diagnostics further down; read those for the current
picture. The noise/batch-size hypothesis here turned out to be secondary.)*

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

## The 20-hour production run (moving reference, `iter_steps=3000`)

Following the fixed-reference tests above, we launched a real (moving-reference)
5-dice production run using the best-evidenced config at the time: η=0.5, lr=1e-5,
β=6, `adv_clip=100` (tightened from the default 1e4, as a hedge against the runaway
updates seen at lr=5e-5), `iter_steps=3000` (up from the `300` that caused run1 to
degrade immediately), NFSP-distilled start, `shaping=0` (turned off to remove a
confound). Run in two segments totalling ~16.6h wall time and 44,000 steps / ~15
outer iterations: `checkpoints/rnad_d5_prod10h/` (steps 0-20,000, ~10h) then a
`--resume` continuation (steps 20,000-44,000, ~6.6h — finished faster than its 10.3h
cap since the machine had no contention overnight).

**Within-iteration convergence now works cleanly** (validating the `iter_steps=3000`
fix): `kl`/`ent`/`gn` all flattened within iteration 0 by step ~2500-3000, unlike
every earlier attempt.

**But win rate vs SmartBot/BestBot still declined steadily through the first run**
(62.3%/49.8% distilled start → 21.0%/30.5% at step 20,000), **then flattened into a
noisy plateau for the entire continuation** rather than continuing to decline or
recovering: steps 20,500-29,500 (iterations 6-9) averaged ~23.6%/27.8% vs
SmartBot/BestBot; steps 30,000-44,000 (iterations 10-14) averaged ~26.8%/29.0% — a
mild (~3pp) uptick, not a reversal, and still far below the distilled starting point.
Win rate vs NFSP stayed a noisy band the whole 44,000 steps (35-49%, slight upward
drift late: ~41% → ~43% average across the same split), never clearly above its
47.2% starting value.

**Successor-vs-predecessor tournament** (playing each outer-iteration checkpoint
head-to-head against the one immediately before it, 400 games, no external bots):
results clustered tightly around 50% (46.5%-53.8% across 7 consecutive pairs,
average 50.0%). Initially read as "surprisingly not degrading" — but on reflection
this is not actually evidence of anything going right or wrong. **R-NaD's
convergence guarantee is about the *sequence's* exploitability trending toward the
original game's equilibrium, not about each iterate beating its immediate
predecessor under the original (unregularized) game.** Each iterate is only the
Nash equilibrium of its own *regularized* sub-game (different payoffs, due to the
KL penalty), so there's no theoretical reason to expect pairwise dominance —
especially in a game plausibly having non-transitive structure. A near-50% result
here is a *stability guardrail pass* (rules out catastrophic divergence), not
evidence of convergence toward equilibrium. The signal that actually matters (the
sequence's exploitability under the *original* game) is only directly measurable at
1 die; at 5 dice we've been inferring it from proxies (external bot win rate, `kl`,
`ent`) that this session's discussion concluded are each imperfect stand-ins.

**Best checkpoint of the whole 44,000-step run remains `rnad_step500.pt`**
(57.0%/54.5%/53.2%, ~23 minutes in) — every later checkpoint, including the final
one, is worse against SmartBot/BestBot than that early point.

## How DeepMind's R-NaD works, and where our implementation differed

Sources: the DeepNash paper (Table 2 and the method section), DeepMind's reference
implementation (`open_spiel/python/algorithms/rnad/rnad.py` at commit 894710bb; removed
from OpenSpiel 2.x wheels), its README and its published Leduc NashConv plot.

**Algorithm.** Outer loop: fix a regularization policy `pi_reg`; the acting player pays
`eta * log(pi/pi_reg)` and the other player receives it, which makes the game strictly
monotone with a unique equilibrium. After `Δm` learner steps the slow target network
becomes the next `pi_reg`; for the first half of each iteration the penalty blends the
newest and previous reference (`alpha = min(1, 2n/Δm)`). Inner loop per step: play a
batch of whole games, two-player v-trace backward over each game (no bootstrapping)
for value targets and per-action Q estimates, NeuRD loss on the logits, regression on
the value head, then move the target network by EMA.

**Paper settings (Stratego):** eta 0.2, lr 5e-5, NeuRD beta 2, NeuRD clip 10,000,
gradient clip 10,000, target EMA 0.001, Adam (0, 0.999), batch 768 games per learner x
768 learner machines (gradients averaged), Δm 10k for the first 100 iterations, 100k for
iterations 101-165, then 35k; 7.21M steps (~170 outer iterations); fine-tuning
threshold 0.03 + discretization 1/32.

**Reference README settings for their Leduc result:** batch 512, lr 5e-5, Δm 50,000,
MLP (256, 256) on the information-state tensor, up to 7M steps, 5 seeds. Their plot
drops fast, bumps up (~0.4 -> ~0.65 around 250k steps), then **plateaus at NashConv
~0.17-0.2 from ~1M to 7M steps** — DeepMind's own sampled R-NaD does not reach zero on a
small game either. A 2026 single-author preprint (GARIP, arXiv 2606.22688) reports
tabular R-NaD on Leduc with exact values never below 0.1 (best 0.24) and blames a lag
("staleness") of the reference; treat as supporting evidence, not proof.

**Differences we found (now switchable; `--preset reference` uses the reference column):**

| Area | Reference | Ours before |
|---|---|---|
| Penalty in the reward stream | expected KL `sum_a pi(a) log(pi/pi_reg)(a)` | sampled log-ratio of the taken action (matches the paper's equations; noisier) |
| Loss normalization | per player, summed | one mean over all rows |
| Clipping | gradient clip 10k after Adam (rarely binds) | grad-norm clip 10 before Adam |
| v-trace rho | `inf` | 1 (irrelevant while on-policy) |
| Target EMA | 0.001 (code; 0.01 in their Leduc README) | 0.01 |
| Δm | list of sizes/repeats (`EntropySchedule`) | one fixed `iter_steps` |
| Network | MLP 256x256 on the full information set | face-shared net on the last 8 bids |

Our v-trace, Q estimate and NeuRD force already matched the paper's equations.

## Phase 1: exact diagnostics on the 1-die game

Reproduce with `diag_d1.py`. CFR+ reference: NashConv 0.00911 (400 it), 0.00441 (1000),
**0.00255 (2000)**, seat-0 value -0.021.

- **Fine-tuning helps a little.** Our best 1-die checkpoint (`rnad_d1_bigbatch`):
  raw 0.405, threshold 0.03 -> 0.383, + discretize/32 -> 0.393.
- **The error is on the main line, not rarely visited infosets.** Replacing our policy
  with CFR+ at the 99% of infosets self-play rarely reaches (6.8% of visits): 0.405 ->
  0.375. Replacing it at the 1% most-visited (93% of visits): -> 0.059; at 3.2% (98.7%
  of visits): -> 0.016.
- **Exact R-NaD converges** (tabular, exact counterfactual values, mirror-ascent step
  0.5, per-infoset steps). eta 0.2: the first fixed point (uniform reference) is
  **0.364** — exactly where all our sampled runs stalled — then 0.100 (iter 2), 0.055
  (5), 0.045 (10), 0.021 (20), 0.016 (40); with only 50 inner steps per iteration it
  reaches **0.0074 after 300 iterations**. Short inner loops are about as good per outer
  iteration as fully converged ones, and far cheaper.
- **Visit-weighted exact R-NaD stalls.** Sampled on-policy training updates each
  infoset in proportion to how often self-play visits it. Giving the exact oracle that
  weighting: 50 inner steps -> stuck at **0.50**; 200 inner steps -> 0.246 after 300
  iterations. This is the main structural weakness of sampled R-NaD here.
- **Our original face-shared network is a floor.** Distilling the exact CFR+ policy
  into it by supervised learning plateaus at NashConv **0.25-0.29** (64 or 128 wide).
  The information-set MLP (`InfoSetNet`) reaches 0.047 after 3,000 steps and is still
  falling. Relevant to any future 5-dice R-NaD run with the face-shared net.

## Reference-config training at 1 die (before the centering fix)

`train_rnad.py --dice 1 --preset reference` as it was then (InfoSetNet, expected-KL penalty,
per-player loss, lr 5e-5, eta 0.2, beta 2, clips 1e4, target EMA 0.001, batch 512), ~0.03s/step,
**still with legal-action NeuRD centering**. Exact NashConv (raw):

| Run | Δm | best | final |
|---|---|---|---|
| A: reference | 50k | 0.594 @ 80k | 0.901 @ 1M (flat ~0.89 from 250k) |
| B: reference | 10k | 0.524 @ 30k | 0.648 @ 50k, stopped |
| C: reference, lr 2e-4 | 10k | 0.639 @ 20k | 0.903 @ 50k, stopped |
| D: reference + `--loss-norm infoset` | 50k | **0.482 @ 100k** | 0.723 @ 1M |
| E: reference + `--loss-norm infoset` | 10k | 0.706 @ 10k | 0.868 @ 50k, stopped |

`--loss-norm infoset` averages the policy loss per distinct information set in the
batch (every visited infoset takes an equal step) — the sampled analogue of the exact
oracle's per-infoset steps. It helps a little (D vs A) but does not stop the
degradation. Every sampled run with Δm 10k got worse with each reference reset, the
same pattern as the 5-dice runs; the exact oracle never degrades, so the likely
cause is noise frozen into each new reference. **Superseded:** A and D never recovered,
and the real cause turned out to be the NeuRD centering (next section).

## Reference comparison: the NeuRD centering bug

Tools in `reference_check/` (see its README): DeepMind's reference `rnad.py`, our 1-die rules
registered as an OpenSpiel game, and a side-by-side loss check.

**Our game as an OpenSpiel game is exact.** OpenSpiel's independent tools give the same
numbers as `exact_d1.py` to 6 decimals: uniform NashConv 1.563371, our CFR+ policy 0.002546,
seat-0 value -0.021200. (OpenSpiel's own `liars_dice` differs only in that 6s are wild and 6
is the highest face; ours has 1s wild and 1 the lowest face.)

**DeepMind's code does not degrade, on either game.** Same settings as our runs (batch 512,
lr 5e-5, eta 0.2, MLP 256x256), exact NashConv raw [fine-tuned]:

| Run | Game | Δm | 10k | 50k | 100k | final |
|---|---|---|---|---|---|---|
| R2 | OpenSpiel liars_dice | 10k | 0.616 | 0.434 | 0.363 | 0.363 [0.266] @ 100k |
| R1 | OpenSpiel liars_dice | 50k | 0.624 | 0.525 | 0.380 | 0.273 [0.166] @ 300k |
| O2 | ours | 10k | 0.605 | 0.363 | **0.258** [0.188] | 0.258 @ 100k |
| O1 | ours | 50k | 0.606 | 0.452 | 0.299 | **0.226 [0.132]** @ 250k |

So the rules were not the problem; our implementation was.

**Side-by-side loss check** (`reference_check/diff_*.py`): one batch from our engine through
DeepMind's `v_trace` / `get_loss_v` / `get_loss_nerd` and through our training helpers, in
float64. Value targets and advantages agree to ~1e-15. Exactly two loss differences:

- our value loss is `0.5 * (v - target)^2`, theirs `(v - target)^2` (value gradient 2x);
- NeuRD centering: ours subtracted the mean of the legal logits, theirs subtracts the sum of
  the legal logits divided by the number of ALL actions. With all-action centering our logit
  gradient matches theirs to 5e-17.

Reading the rest of the code found four more differences: weight init (Haiku truncated
normal, std 1/sqrt(fan_in), zero bias), Adam eps (1e-7), the initial regularization policy
(the initial network, not uniform), and the input layout (same information).

**Bisect** (our code, our game, Δm 10k, seed 5): all six matched, then each reverted alone.
Raw NashConv:

| Run | 20k | 40k | 60k | entropy @ 60k |
|---|---|---|---|---|
| DeepMind's code (O2) | 0.534 | 0.415 | 0.332 | |
| ours, all matched | 0.531 | 0.426 | **0.358** | 1.05 |
| reverted: centering (legal mean) | **0.452** | 0.627 | **0.733** | 0.85 |
| reverted: value-loss factor | 0.534 | 0.417 | 0.343 | 1.04 |
| reverted: weight init | 0.594 | 0.445 | 0.357 | 1.00 |
| reverted: initial reg policy | 0.536 | 0.438 | 0.365 | 1.05 |
| reverted: Adam eps | 0.537 | 0.417 | 0.348 | 1.04 |

The centering alone reproduces the old failure (early gains, then degradation with entropy
collapse after a couple of resets); the other four differences don't matter. Why it matters
(hypothesis, not proven): at a single state both versions move the policy identically; they
differ in the beta threshold check. All-action centering also limits the overall level of the
legal logits; legal-mean centering lets all logits drift together, which in a shared network
can leak into other states.

`--preset reference` now sets `center=all`, `value_coef=2`, `adam_eps=1e-7`, `init=haiku`,
`init_reg=net`. Config defaults keep the old behaviour so old checkpoints resume unchanged.

**Even the reference doesn't reach Nash here:** it flattens around 0.22-0.26 raw (0.13 with
fine-tuning) after 100-250k steps, versus CFR+ 0.0026 and the exact oracle 0.0074. That
matches DeepMind's own Leduc plateau (~0.18) and is the realistic target for sampled R-NaD.

## Network floor and the liar decision (1 die, distilled from CFR+)

`diag_d1.py floor` trains a network directly on the exact CFR+ policy (all 24,576 infosets,
full batch, 3,000 steps) and measures its NashConv — what the architecture can represent.
`diag_d1.py liar` replaces only P(liar), or only the choice among raises, with CFR+'s.

| Network | NashConv | fix only P(liar) | fix only raises | avg P(liar) error |
|---|---|---|---|---|
| face-shared net (`face`) | 0.283 | 0.151 | 0.163 | 0.067 |
| + face rank (`--face-rank`) | 0.147 | 0.108 | 0.075 | 0.038 |
| + round context MLP (`--info-ctx 64`) | 0.130 | 0.093 | 0.071 | 0.025 |
| + both | 0.113 | 0.109 | **0.030** | 0.014 |
| `InfoSetNet` | **0.043** | 0.037 | 0.023 | 0.013 |

The face-shared net's floor (~0.28) is mostly fixed by telling the shared face MLP each face's
rank in the bid order (the only thing distinguishing faces 2-6) and/or adding an MLP over the
exact round encoding to the context. Rank weights are zero-initialized so the net starts
exactly face-invariant. With both, the liar decision is as good as `InfoSetNet`'s; the
remaining gap is in choosing among raises. Single seed; ~±0.02 noise between checkpoints.

## Suggested next steps

1. **Re-run 5 dice with the fixed settings** (`--preset reference`, i.e. all-action
   centering). Every 5-dice conclusion in this doc predates the fix.
2. **For 5 dice, compare `InfoSetNet` with the face-shared net + `--face-rank --info-ctx 64`**:
   the face net's weight sharing is a sample-efficiency bet that only pays off at 5 dice.
3. If R-NaD still plateaus too high: a smoothed reference (GARIP / EMAgnet), or regularized
   PPO (see the ICLR 2026 note in the PPO section).
4. Lower priority: an approximate best-response probe for 5-dice comparisons.

## Checkpoints produced this session (gitignored, not in this PR)

`checkpoints/rnad_d5_prod10h/` (the 44,000-step moving-reference production run,
best checkpoint is `rnad_step500.pt`, not the final one — see above),
`checkpoints/rnad_d1_bigbatch/` (1 die, 32,768 games/step: NashConv 0.395-0.43),
`checkpoints/rnad_d1_ref_*` (the reference-config 1-die runs above),
`checkpoints/rnad_d5_run1/`, `checkpoints/rnad_d1_fixedref/`,
`checkpoints/rnad_d5_fixedref/`, `checkpoints/rnad_d5_fixedref_5e5/` — all local only
(`checkpoints/` is gitignored). Share these separately if the next machine needs them
instead of retraining; otherwise the distillation warm-start makes re-establishing a
comparable starting point cheap (~3 min) as long as
`checkpoints/shared_face/nfsp_latest.pt` is available.
