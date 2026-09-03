#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main
echo "GPU1: Restormer FiLM -> Gated -> ASF | Start: $(date)"
CUDA_VISIBLE_DEVICES=1 python train_compare_models.py --model shadow_guided_restormer_film --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/sgfm_lol/
echo "[1/3] Restormer+FiLM done: $(date)"
CUDA_VISIBLE_DEVICES=1 python train_compare_models.py --model shadow_guided_restormer_gated --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/sggf_lol/
echo "[2/3] Restormer+Gated done: $(date)"
CUDA_VISIBLE_DEVICES=1 python train_compare_models.py --model shadow_guided_restormer_asf --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/sgasf_lol/
echo "[3/3] Restormer+ASF done: $(date)"
echo "GPU1 chain complete: $(date)"
