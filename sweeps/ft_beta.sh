#!/bin/bash
# Training-time fine-tuning vs a larger NeuRD beta, 1 die (2026-10-08). All four runs resume the batch-4096,
# Δm-50k runs of sweeps/b4096_long.sh at step 150k (already near the plateau, ~0.23 raw / 0.12 fine-tuned), so
# they share that history and the original runs' 150k->315k stretch is the control.
#   ft:     DeepNash fine-tuning from step 150k (v-trace evaluates the thresholded + discretized policy), seeds 1, 2
#   beta4, beta8: NeuRD logit threshold raised from 2 to 4 / 8 at step 150k, seed 1
# ~0.084 s/step with 4 runs -> ~330k more steps (to ~480k) in 7.8h.
# Summarize with: python3 sweeps/summarize_sweep.py checkpoints/ft_beta/logs
cd "$(dirname "$0")/.."
OUT=checkpoints/ft_beta
SRC=checkpoints/b4096_long
mkdir -p $OUT/logs
COMMON="--threads 1 --max-hours 7.8 --total-steps 1000000 --log-every 5000 --eval-every 5000 --save-every 50000"
for seed in 1 2; do
  nohup python3 train_rnad.py $COMMON --resume $SRC/dm50k_s$seed/rnad_step150000.pt --finetune-from 150000 --seed $seed \
    --checkpoint-dir $OUT/ft_s$seed > $OUT/logs/ft_s$seed.log 2>&1 &
done
for b in 4 8; do
  nohup python3 train_rnad.py $COMMON --resume $SRC/dm50k_s1/rnad_step150000.pt --beta $b --seed 1 \
    --checkpoint-dir $OUT/beta${b}_s1 > $OUT/logs/beta${b}_s1.log 2>&1 &
done
echo "launched 4 runs"
