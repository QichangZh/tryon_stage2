#!/bin/bash
#SBATCH --job-name=accelerate_stage2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2        # 改回4，使用4个MIG实例
#SBATCH --cpus-per-task=60
#SBATCH --mem=800G
#SBATCH --gres=gpu:h100-96:2       # 申请4个MIG实例
#SBATCH --time=4-01:00:00
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qiczhang@163.com
##SBATCH --nodelist=xgph23

source ~/.bashrc
conda activate tryon

# 使用所有可用的MIG实例启动训练
accelerate launch --num_machines 1,--gpu_ids 0,1 --num_processes 2 --use_deepspeed --mixed_precision="bf16"  stage2_train_inpaint_model.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
  --img_height=1024  \
  --img_width=768   \
  --learning_rate=1e-4 \
  --train_batch_size=8 \
  --val_batch_size=8 \
  --resume_from_checkpoint="logs/stage2" \
  --max_train_steps=1000000 \
  --mixed_precision="bf16" \
  --checkpointing_steps=50  \
  --noise_offset=0.1 \
  --lr_warmup_steps=5000  \
  --seed=42
