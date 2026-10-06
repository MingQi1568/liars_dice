#!/bin/bash
# Overnight 1-die R-NaD sweep (2026-10-06). One lever at a time from the baseline (reference defaults:
# batch 512, lr 5e-5 constant, target EMA 0.001, eta 0.2) with delta_m 10k; 2 seeds each.
# Exact NashConv every eval (raw and fine-tuned). Outputs: checkpoints/sweep_d1/<arm>_s<seed>/ and logs/.
# Summarize with: python3 sweeps/summarize_sweep.py checkpoints/sweep_d1/logs
cd "$(dirname "$0")/.."
OUT=checkpoints/sweep_d1
mkdir -p $OUT/logs
COMMON="--dice 1 --threads 1 --max-hours 7.8"
SMALL="--games-per-step 512 --total-steps 1000000 --log-every 10000 --eval-every 10000 --save-every 100000"
BIG="--games-per-step 4096 --total-steps 350000 --log-every 5000 --eval-every 5000 --save-every 50000"
declare -a ARMS=(
  "base|$SMALL --sched-sizes 10000"
  "dm50k|$SMALL --sched-sizes 50000"
  "lrdecay|$SMALL --sched-sizes 10000 --lr-points 500000:5e-5,1000000:5e-6"
  "gamma01|$SMALL --sched-sizes 10000 --gamma 0.01"
  "eta05|$SMALL --sched-sizes 10000 --eta 0.5"
  "b4096|$BIG --sched-sizes 10000"
  "b4096_lrdecay|$BIG --sched-sizes 10000 --lr-points 175000:5e-5,350000:5e-6"
)
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; flags="${arm#*|}"
  for seed in 1 2; do
    nohup python3 train_rnad.py $COMMON $flags --seed $seed --checkpoint-dir $OUT/${name}_s${seed} \
      > $OUT/logs/${name}_s${seed}.log 2>&1 &
  done
done
echo "launched $(( ${#ARMS[@]} * 2 )) runs"
