#!/bin/bash
# 12-hour runs at 1 die (2026-10-12), 4 runs on the 4 performance cores, seed 1 each:
#   b4096_dm50k   the batch-4096 / Δm-50k control (sweeps/b4096_long.sh dm50k_s1) continued from step ~319k
#   b16384_dm50k  batch 4x4096, Δm 50k, from scratch
#   b16384_dm10k  batch 4x4096, Δm 10k, from scratch
#   cfr_init      batch 4096, Δm 50k, starting from the CFR+ equilibrium (init_cfr_d1.py; NashConv 0.038 at
#                 step 0): does self-play keep it near equilibrium or drift away?
# Measured with 4 runs: ~0.084 s/step at batch 4096 (~500k steps in 12h), ~0.35 s/step at batch 16384 (~120k).
cd "$(dirname "$0")/.."
OUT=checkpoints/long12h
mkdir -p $OUT/logs
COMMON="--threads 1 --max-hours 12 --total-steps 3000000"
B4="--log-every 5000 --eval-every 5000 --save-every 50000"
B16="--games-per-step 16384 --log-every 2500 --eval-every 2500 --save-every 25000 --dice 1 --seed 1"
nohup python3 train_rnad.py $COMMON $B4 --seed 1 --resume checkpoints/b4096_long/dm50k_s1/rnad_latest.pt \
  --checkpoint-dir $OUT/b4096_dm50k > $OUT/logs/b4096_dm50k.log 2>&1 &
nohup python3 train_rnad.py $COMMON $B16 --sched-sizes 50000 --checkpoint-dir $OUT/b16384_dm50k > $OUT/logs/b16384_dm50k.log 2>&1 &
nohup python3 train_rnad.py $COMMON $B16 --sched-sizes 10000 --checkpoint-dir $OUT/b16384_dm10k > $OUT/logs/b16384_dm10k.log 2>&1 &
nohup python3 train_rnad.py $COMMON $B4 --seed 1 --resume checkpoints/cfr_init/rnad_cfr_step0.pt \
  --checkpoint-dir $OUT/cfr_init > $OUT/logs/cfr_init.log 2>&1 &
echo "launched 4 runs"
