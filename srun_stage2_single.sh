#!/bin/bash
#SBATCH --job-name=accelerate_stage2       # 修改为stage2
#SBATCH --nodes=1                          # 需要 2 个节点
#SBATCH --ntasks-per-node=4                # 每个节点启动 2 个任务（与GPU数量匹配）
#SBATCH --cpus-per-task=30                 # 每个任务使用 30 个 CPU 核心
#SBATCH --mem=800G                         # 每个节点分配 400GB 内存
#SBATCH --gres=gpu:h100-47:4               # 每个节点分配 2 块 GPU
#SBATCH --time=4-01:00:00                  # 最长运行时间
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL               # 何时给用户发邮件
#SBATCH --mail-user=qiczhang@163.com       # 接收邮件的地址
##SBATCH --nodelist=xgph12,xgph13           # 指定节点 (根据实际情况修改)

source ~/.bashrc
conda activate tryon

# 在source ~/.bashrc和conda activate后添加
echo "======= 检查可用的CUDA设备 ======="
nvidia-smi --list-gpus
echo "======= GPU详细信息 ======="
nvidia-smi
echo "======= CUDA_VISIBLE_DEVICES环境变量 ======="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "======= 检测到的CUDA设备数量 ======="
python -c "import torch; print(f'PyTorch检测到的GPU数量: {torch.cuda.device_count()}')"
python -c "import torch; print(f'可用的GPU设备: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')"



accelerate launch --num_machines 1 --gpu_ids 0,1,2,3 --num_processes 4 --use_deepspeed --mixed_precision="bf16"  stage2_train_inpaint_model.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
  --img_height=1024  \
  --img_width=768   \
  --learning_rate=1e-4 \
  --train_batch_size=8 \
  --val_batch_size=32 \
  --resume_from_checkpoint="logs/stage2" \
  --max_train_steps=1000000 \
  --mixed_precision="bf16" \
  --checkpointing_steps=50  \
  --noise_offset=0.1 \
  --lr_warmup_steps 5000  \
  --seed 42
