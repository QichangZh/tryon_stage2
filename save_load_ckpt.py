import os
import logging
from typing import Optional, Dict, Any
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def checkpoint_model(
    checkpoint_folder: str,
    ckpt_id: int,
    model: nn.Module,
    epoch: int,
    last_global_step: int,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    **kwargs: Any
) -> None:
    """保存检查点，包括模型权重和训练状态
    
    Args:
        checkpoint_folder: 检查点保存目录
        ckpt_id: 检查点ID
        model: PyTorch模型
        epoch: 当前训练轮次
        last_global_step: 当前全局训练步数
        optimizer: 优化器 (可选)
        lr_scheduler: 学习率调度器 (可选) 
        **kwargs: 其他需要保存的参数
    """
    os.makedirs(checkpoint_folder, exist_ok=True)
    save_path = os.path.join(checkpoint_folder, f"checkpoint-{ckpt_id}.pt")
    
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
        "state_dict": model.state_dict(),  # 保存模型权重
    }
    
    # 保存优化器状态（如果提供）
    if optimizer is not None:
        checkpoint_state_dict["optimizer_state"] = optimizer.state_dict()
    
    # 保存学习率调度器状态（如果提供）
    if lr_scheduler is not None:
        checkpoint_state_dict["lr_scheduler_state"] = lr_scheduler.state_dict()
        
    # 添加其他额外参数
    checkpoint_state_dict.update(kwargs)
    
    # 保存检查点
    try:
        torch.save(checkpoint_state_dict, save_path)
        logging.info(f"Successfully saved checkpoint to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {save_path}: {str(e)}")
        raise


def load_training_checkpoint(
    model: nn.Module,
    load_dir: str,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    ckpt_id: Optional[int] = None,
    **kwargs: Any
) -> tuple[nn.Module, int, int]:
    """从检查点加载模型和训练状态
    
    Args:
        model: PyTorch模型
        load_dir: 检查点加载目录
        optimizer: 优化器 (可选)
        lr_scheduler: 学习率调度器 (可选)
        ckpt_id: 指定要加载的检查点ID (可选,默认加载最新的检查点)
        
    Returns:
        tuple: (加载后的模型, 当前轮次, 当前全局步数)
    """
    # 确保目录存在
    if not os.path.exists(load_dir):
        logging.info(f"Checkpoint directory {load_dir} not found, starting from step 0")
        return model, 0, 0
    
    # 获取所有.pt检查点文件
    checkpoint_files = [f for f in os.listdir(load_dir) if f.startswith('checkpoint-') and f.endswith('.pt')]
    
    # 如果没有检查点文件，从头开始训练
    if not checkpoint_files:
        logging.info(f"No checkpoint files found in {load_dir}, starting from step 0")
        return model, 0, 0
    
    try:
        if ckpt_id is not None:
            # 加载指定ID的检查点
            checkpoint_path = os.path.join(load_dir, f"checkpoint-{ckpt_id}.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        else:
            # 加载最新的检查点
            steps = [int(f.split('-')[-1].replace('.pt', '')) for f in checkpoint_files]
            latest_step = max(steps)
            checkpoint_path = os.path.join(load_dir, f"checkpoint-{latest_step}.pt")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        # 加载检查点
        checkpoint_state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # 处理模型权重
        state_dict = checkpoint_state_dict.get("state_dict") or checkpoint_state_dict.get("module")
        if state_dict is None:
            raise KeyError("Neither 'state_dict' nor 'module' key found in checkpoint")
            
        # 处理DataParallel保存的模型权重
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        
        # 加载优化器状态（如果有）
        if optimizer is not None and "optimizer_state" in checkpoint_state_dict:
            optimizer.load_state_dict(checkpoint_state_dict["optimizer_state"])
        
        # 加载学习率调度器状态（如果有）
        if lr_scheduler is not None and "lr_scheduler_state" in checkpoint_state_dict:
            lr_scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler_state"])
        
        epoch = checkpoint_state_dict.get("epoch", 0)
        last_global_step = checkpoint_state_dict.get("last_global_step", 0)
        
        return model, epoch, last_global_step
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {str(e)}")
        raise