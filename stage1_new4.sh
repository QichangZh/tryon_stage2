#!/bin/bash
#SBATCH --job-name=accelerate_stage1       # 作业名称
#SBATCH --nodes=2                          # 需要 2 个节点
#SBATCH --ntasks-per-node=2                # 每个节点启动 2 个任务（与GPU数量匹配）
#SBATCH --cpus-per-task=30                 # 每个任务使用 30 个 CPU 核心
#SBATCH --mem=400G                         # 每个节点分配 400GB 内存
#SBATCH --gres=gpu:h100-47:2               # 每个节点分配 2 块 GPU
#SBATCH --time=4-01:00:00                    # 最长运行时间
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL               # 何时给用户发邮件
#SBATCH --mail-user=qiczhang@163.com       # 接收邮件的地址
#SBATCH --nodelist=xgpi14,xgpi16           # 指定节点 (根据实际情况修改)

# (可选) 如果需要 conda 环境，先加载相应模块再激活环境
source ~/.bashrc
conda activate tryon
# pixel!!!!
# 打印当前工作目录（可选）
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
WORKER_GPU=4
WORKER_NUM=2
# 直接让 Slurm 在每个节点/进程上通过 srun 启动任务
# Accelerate 会自动检测本地的进程/设备信息完成分布式
accelerate launch  --num_processes $WORKER_GPU --num_machines $WORKER_NUM --use_deepspeed --main_process_ip $head_node --main_process_port 29555 \
    stage1_train_prior_model.py \
    --pretrained_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
    --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --img_path="./data/VTON/train" \
    --output_dir="output_dir" \
    --img_height=512 \
    --img_width=512 \
    --train_batch_size=128 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100010 \
    --noise_offset=0.1 \
    --learning_rate=1e-05 \
    --weight_decay=0.01 \
    --lr_scheduler="constant" --num_warmup_steps=2000 \
    --checkpointing_steps=5000 \
    --resume_from_checkpoint="./output_dir" \
    --seed 42