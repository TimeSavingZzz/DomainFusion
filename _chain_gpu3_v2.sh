#!/bin/bash
# GPU 3 chain: wait for FiLM SD7K → CrossAttn SD7K → Restormer SD7K
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main

TARGET_PID=175
echo "[CHAIN] Waiting for FiLM SD7K (PID ) to finish..."
while kill -0  2>/dev/null; do sleep 30; done
echo "[CHAIN] FiLM SD7K done. Starting CrossAttn SD7K..."

CUDA_VISIBLE_DEVICES=3 python train_compare_models.py   --model shadow_guided_restormer_crossattn   --dataset sd7k --epochs 200 --res 192 --batch_size 1   --output ./experiment_results/sgcr_sd7k/   > /mnt/exp_sgcr_sd7k.log 2>&1

echo "[CHAIN] CrossAttn SD7K done. Starting Restormer SD7K baseline..."

CUDA_VISIBLE_DEVICES=3 python train_compare_models.py   --model restormer   --dataset sd7k --epochs 200 --res 320 --batch_size 1   --output ./experiment_results/restormer_sd7k/   > /mnt/exp_restormer_sd7k.log 2>&1

echo "[CHAIN] GPU 3 queue complete."
