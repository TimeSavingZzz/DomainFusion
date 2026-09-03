#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main
echo "GPU3: NAFNet FiLM -> Gated -> Restormer CrossAttn | Start: $(date)"
CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model shadow_guided_nafnet_film --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nafnet_film_lol/
echo "[1/3] NAFNet+FiLM done: $(date)"
CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model shadow_guided_nafnet_gated --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nafnet_gated_lol/
echo "[2/3] NAFNet+Gated done: $(date)"
CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model shadow_guided_restormer_crossattn --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/sgcr_lol/
echo "[3/3] Restormer+CrossAttn done: $(date)"
echo "GPU3 chain complete: $(date)"
