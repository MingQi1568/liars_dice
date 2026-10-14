#!/bin/bash
# Long batch-4096 test at 1 die (2026-10-07): does the bigger batch, which was ahead of batch 512 at equal outer
# iterations in sweep_d1 but only got 13 of them, end below the ~0.22 plateau? 2 seeds x (Δm 10k, Δm 50k),
# constant lr 5e-5 (comparable to sweep_d1's arms), ~0.084 s/step with 4 runs on this Mac's 4 performance
# cores -> ~330k steps in 7.8h. Summarize with: python3 sweeps/summarize_sweep.py checkpoints/b4096_long/logs
cd "$(dirname "$0")/.."
OUT=checkpoints/b4096_long
mkdir -p $OUT/logs
COMMON="--dice 1 --threads 1 --max-hours 7.8 --games-per-step 4096 --total-steps 1000000 --log-every 5000 --eval-every 5000 --save-every 50000"
for dm in 10000 50000; do
  for seed in 1 2; do
    name=dm$((dm / 1000))k_s$seed
    nohup python3 train_rnad.py $COMMON --sched-sizes $dm --seed $seed --checkpoint-dir $OUT/$name > $OUT/logs/$name.log 2>&1 &
  done
done
echo "launched 4 runs"
