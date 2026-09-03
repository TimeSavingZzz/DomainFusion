#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
cd /mnt/ShaDocFormer-main

echo "[CHAIN] GPU3: FiLM SD7K"
CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model shadow_guided_restormer_film --dataset sd7k --epochs 200 --res 320 --batch_size 1 --output ./experiment_results/sgfm_sd7k/ > /mnt/exp_sgfm_sd7k.log 2>&1
echo "[CHAIN] GPU3: FiLM SD7K DONE, starting Large SD7K"

CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model shadow_guided_restormer_large --dataset sd7k --epochs 200 --res 320 --batch_size 1 --output ./experiment_results/sglarge_sd7k/ > /mnt/exp_sglarge_sd7k.log 2>&1
echo "[CHAIN] GPU3: Large SD7K DONE, starting Restormer SD7K"

CUDA_VISIBLE_DEVICES=3 python train_compare_models.py --model restormer --dataset sd7k --epochs 200 --res 320 --batch_size 1 --output ./experiment_results/restormer_sd7k/ > /mnt/exp_restormer_sd7k.log 2>&1
echo "[CHAIN] GPU3: ALL DONE"
