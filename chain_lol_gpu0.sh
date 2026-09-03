#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main
echo "GPU0: Restormer baseline -> Concat -> Large | Start: $(date)"
CUDA_VISIBLE_DEVICES=0 python train_compare_models.py --model restormer --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/restormer_lol/
echo "[1/3] Restormer baseline done: $(date)"
CUDA_VISIBLE_DEVICES=0 python train_compare_models.py --model shadow_guided_restormer_no_sgca --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nosgca_lol/
echo "[2/3] Restormer+Concat done: $(date)"
CUDA_VISIBLE_DEVICES=0 python train_compare_models.py --model shadow_guided_restormer_large --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/sglarge_lol/
echo "[3/3] Restormer+Large done: $(date)"
echo "GPU0 chain complete: $(date)"
