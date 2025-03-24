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
scontrol show hostnames $SLURM_JOB_NODELIST 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "所有节点: ${nodes[@]}"
nodes_array=(${nodes[@]})
head_node=${nodes_array[0]}
echo "所有节点: ${nodes_array[@]}"
echo "主节点: $head_node"

# # 创建临时目录保存每个节点的输出
# mkdir -p gpu_info
# rm -f gpu_info/*

# # 分别在每个节点上执行GPU检查，并保存到独立文件
# echo "======= 正在检查所有节点的GPU分配情况 ======="
# for node in "${nodes_array[@]}"; do
#     echo "正在检查节点: $node"
#     srun -N1 -n1 --nodelist=$node bash -c "echo '节点 $(hostname) 的GPU信息:' > gpu_info/${node}_info.txt && nvidia-smi >> gpu_info/${node}_info.txt"
#     srun -N1 -n1 --nodelist=$node bash -c "echo '可见GPU: $CUDA_VISIBLE_DEVICES' >> gpu_info/${node}_info.txt"
#     srun -N1 -n1 --nodelist=$node bash -c "echo 'GPU数量: $(nvidia-smi --list-gpus | wc -l)' >> gpu_info/${node}_info.txt"
# done

# # 显示所有节点的GPU信息
# echo "======= 所有节点的GPU信息 ======="
# for node in "${nodes_array[@]}"; do
#     echo "----------------------------------------"
#     echo "节点 $node 的GPU信息:"
#     cat gpu_info/${node}_info.txt
#     echo "----------------------------------------"
# done

# # 显示进程和GPU映射 (在每个节点上分别执行)
# echo "======= 进程和GPU映射信息 ======="
# for node in "${nodes_array[@]}"; do
#     srun -N1 -n1 --nodelist=$node bash -c "echo '节点 $(hostname) 上的进程:' > gpu_info/${node}_proc.txt && ps -ef | grep python | grep -v grep >> gpu_info/${node}_proc.txt"
# done

# # 合并显示所有进程信息
# for node in "${nodes_array[@]}"; do
#     echo "----------------------------------------"
#     echo "节点 $node 的进程信息:"
#     cat gpu_info/${node}_proc.txt
#     echo "----------------------------------------"
# done


# 在适当位置添加（建议在GPU检查部分之后）

echo "======= 检查各节点GPU编号 ======="
for node in "${nodes[@]}"; do
  echo "节点 $node 的GPU编号和分配情况:"
  # 正确输出节点名称和GPU信息
  srun -N1 -n1 --nodelist=$node bash -c "
    echo \"节点 \$(hostname) 的CUDA_VISIBLE_DEVICES:\"
    echo \$CUDA_VISIBLE_DEVICES
    nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader
  "
done

# 在每个节点上显式设置GPU环境变量
for node in "${nodes[@]}"; do
  srun -N1 -n1 --nodelist=$node bash -c "
    export CUDA_VISIBLE_DEVICES=0,1
    echo \"节点 \$(hostname) 设置 CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
  "
done


echo "======= 查找可用端口 ======="
# 在主节点上查找可用端口
available_port=$(srun -N1 -n1 --nodelist=${nodes[0]} bash -c '
    for port in {25000..26000}; do
        if ! (echo >/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
            echo "$port"
            break
        fi
    done
')

if [ -n "$available_port" ]; then
    PORT_TO_CHECK=$available_port
    echo "将使用端口: $PORT_TO_CHECK 进行节点间通信"
else
    PORT_TO_CHECK=25432  # 默认端口
    echo "未找到可用端口，使用默认端口: $PORT_TO_CHECK"
fi

# 修正的accelerate命令行
srun accelerate launch \
    --num_machines 2 \
    --num_processes 4 \
    --main_process_ip $head_node \
    --main_process_port $PORT_TO_CHECK \
    --use_deepspeed \
    --mixed_precision="bf16" \
    stage2_train_inpaint_model.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --image_encoder_p_path='facebook/dinov2-giant' \
    --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
    --img_height=512  \
    --img_width=512   \
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