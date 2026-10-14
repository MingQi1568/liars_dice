#!/bin/bash
# Overnight 2026-10-13, 9h, 1 die, both resumed from sweeps/long12h.sh checkpoints:
#   b16384_dm50k  batch 16384, Δm 50k, continued from step ~151k (its plateau was still open after 3 resets)
#   b4096_dm500k  the batch-4096 control at step ~776k (15 resets of 50k), switched to Δm 500k: the schedule
#                 keeps the 15 finished 50k iterations and the next reset is at 1.25M. Note the new 500k
#                 iteration starts at 750k, so the reward's regularization blend (alpha = 2n/Δm) restarts at
#                 ~0.1 and ramps back to 1 by 1.0M (mostly the 700k reference until then).
cd "$(dirname "$0")/.."
L=checkpoints/long12h
COMMON="--threads 1 --max-hours 9 --total-steps 3000000"
nohup python3 train_rnad.py $COMMON --resume $L/b16384_dm50k/rnad_latest.pt --log-every 2500 --eval-every 2500 \
  --save-every 25000 --checkpoint-dir $L/b16384_dm50k > $L/logs/b16384_dm50k_cont2.log 2>&1 &
nohup python3 train_rnad.py $COMMON --resume $L/b4096_dm50k/rnad_latest.pt --sched-sizes 50000 500000 --sched-repeats 15 1 \
  --log-every 5000 --eval-every 5000 --save-every 50000 --checkpoint-dir $L/b4096_dm500k > $L/logs/b4096_dm500k.log 2>&1 &
echo "launched 2 runs"
