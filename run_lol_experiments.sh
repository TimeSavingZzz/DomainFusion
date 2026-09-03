#!/bin/bash
# LOL-v1 Fusion Strategy Experiments
# Launches training for all model variants on LOL dataset
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main

EPOCHS=200
LR=2e-4
RES=256
DATASET=lol

echo "=== LOL-v1 Fusion Strategy Experiments ==="
echo "Date: $(date)"
echo ""

# Phase 1: Restormer baselines + fusion variants
echo "--- Phase 1: Restormer variants ---"
for model in restormer shadow_guided_restormer_no_sgca shadow_guided_restormer_film shadow_guided_restormer_gated shadow_guided_restormer_asf; do
    echo "Launching $model on $DATASET..."
    nohup python train_compare_models.py --model $model --dataset lol --epochs $EPOCHS --lr $LR --res $RES --output ./experiment_results/${model}_lol > /mnt/log_${model}_lol.txt 2>&1 &
    echo "  PID: $!"
    sleep 5  # stagger launches
done

echo ""
echo "--- Phase 2: NAFNet variants ---"
for model in nafnet shadow_guided shadow_guided_no_sgca shadow_guided_nafnet_film shadow_guided_nafnet_gated shadow_guided_nafnet_asf; do
    echo "Launching $model on $DATASET..."
    nohup python train_compare_models.py --model $model --dataset lol --epochs $EPOCHS --lr $LR --res $RES --output ./experiment_results/${model}_lol > /mnt/log_${model}_lol.txt 2>&1 &
    echo "  PID: $!"
    sleep 5
done

echo ""
echo "All launched. Use 'nvidia-smi' to monitor GPUs."
echo "Logs: /mnt/log_*_lol.txt"
