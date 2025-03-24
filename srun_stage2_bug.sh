#!/bin/bash
#SBATCH --job-name=accelerate_stage2       # 修改为stage2
#SBATCH --nodes=1                          # 需要1个节点
#SBATCH --ntasks-per-node=2                # 保持原有的4个任务配置
#SBATCH --cpus-per-task=60                 # 每个任务使用30个CPU核心
#SBATCH --mem=800G                         # 每个节点分配800GB内存
#SBATCH --gres=gpu:h100-47:4               # 保持申请4块MIG实例不变
#SBATCH --time=4-01:00:00                  # 最长运行时间
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL               # 何时给用户发邮件
#SBATCH --mail-user=qiczhang@163.com       # 接收邮件的地址
#SBATCH --nodelist=xgph23           # 指定节点 (根据实际情况修改)

source ~/.bashrc
conda activate tryon

# 清除任何CUDA_VISIBLE_DEVICES设置，让PyTorch能够看到系统识别的完整GPU
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1

# 验证识别的GPU情况
echo "======= 验证GPU可见性 ======="
nvidia-smi
python -c "import torch; print(f'PyTorch检测到的GPU数量: {torch.cuda.device_count()}')"

# 使用PyTorch识别的两个完整H100 GPU启动训练
accelerate launch --num_machines 1 --gpu_ids 0,1 --num_processes 2 --use_deepspeed --mixed_precision="bf16" stage2_train_inpaint_model.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
  --img_height=1024  \
  --img_width=768   \
  --learning_rate=1e-4 \
  --train_batch_size=16 \
  --val_batch_size=32 \
  --resume_from_checkpoint="logs/stage2" \
  --max_train_steps=1000000 \
  --mixed_precision="bf16" \
  --checkpointing_steps=50  \
  --noise_offset=0.1 \
  --lr_warmup_steps=5000  \
  --seed=42
