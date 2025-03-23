#!/bin/bash
#SBATCH --job-name=accelerate_stage2       # 修改为stage2
#SBATCH --nodes=2                          # 需要 2 个节点
#SBATCH --ntasks-per-node=2                # 每个节点启动 2 个任务（与GPU数量匹配）
#SBATCH --cpus-per-task=30                 # 每个任务使用 30 个 CPU 核心
#SBATCH --mem=200G                         # 每个节点分配 400GB 内存
#SBATCH --gres=gpu:a100-40:2               # 每个节点分配 2 块 GPU
#SBATCH --time=4-01:00:00                  # 最长运行时间
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL               # 何时给用户发邮件
#SBATCH --mail-user=qiczhang@163.com       # 接收邮件的地址
##SBATCH --nodelist=xgph12,xgph13           # 指定节点 (根据实际情况修改)

# (可选) 如果需要 conda 环境，先加载相应模块再激活环境
source ~/.bashrc
conda activate tryon

# 打印当前工作目录
pwd
scontrol show hostnames $SLURM_JOB_NODELIS 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "${nodes_array[@]}"

# 修正的accelerate命令行
srun accelerate launch \
    --num_machines 2 \
    --num_processes 4 \
    --main_process_ip $head_node \
    --main_process_port 29555 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    --mixed_precision="bf16" \
    stage2_train_inpaint_model.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --image_encoder_p_path='facebook/dinov2-giant' \
    --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
    --img_height=512  \
    --img_width=512   \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --val_batch_size=32 \
    --resume_from_checkpoint="logs/stage2" \
    --max_train_steps=1000000 \
    --mixed_precision="bf16" \
    --checkpointing_steps=50  \
    --noise_offset=0.1 \
    --lr_warmup_steps 5000  \
    --seed 42