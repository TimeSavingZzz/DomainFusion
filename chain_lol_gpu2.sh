#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main
echo "GPU2: NAFNet baseline -> Concat -> ASF | Start: $(date)"
CUDA_VISIBLE_DEVICES=2 python train_compare_models.py --model nafnet --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nafnet_lol/
echo "[1/3] NAFNet baseline done: $(date)"
CUDA_VISIBLE_DEVICES=2 python train_compare_models.py --model shadow_guided_no_sgca --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nafnet_nosgca_lol/
echo "[2/3] NAFNet+Concat done: $(date)"
CUDA_VISIBLE_DEVICES=2 python train_compare_models.py --model shadow_guided_nafnet_asf --dataset lol --epochs 200 --lr 2e-4 --res 256 --output ./experiment_results/nafnet_asf_lol/
echo "[3/3] NAFNet+ASF done: $(date)"
echo "GPU2 chain complete: $(date)"
