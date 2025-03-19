import os
import logging
import torch
import accelerate  # 添加这个导入到文件顶部




def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, optimizer=None, lr_scheduler=None,  **kwargs):
    """保存模型检查点，包含模型权重、优化器状态、学习率调节器状态等"""
    os.makedirs(checkpoint_folder, exist_ok=True)
    save_path = os.path.join(checkpoint_folder, f"checkpoint-{ckpt_id}")
    
    # 准备保存的状态
    state_dict = {
        "module": model.state_dict(),  # 模型权重
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    
    # 如果有优化器，保存其状态
    if optimizer is not None:
        logging.info(f"Saving optimizer state, optimizer type: {type(optimizer)}")
        try:
            if isinstance(optimizer, accelerate.utils.deepspeed.DeepSpeedOptimizerWrapper):
                # 获取完整的 DeepSpeed 状态
                ds_state = optimizer.optimizer.state_dict()
                state_dict["optimizer_state"] = {
                    "ds_state": ds_state,  # DeepSpeed 完整状态
                    "base_state": optimizer.optimizer.base_optimizer_state  # 基础优化器状态
                }
            else:
                state_dict["optimizer_state"] = optimizer.state_dict()
            
            logging.info(f"Optimizer state keys: {state_dict['optimizer_state'].keys()}")
        except Exception as e:
            logging.error(f"Failed to save optimizer state: {e}")
    
    
    # 如果有学习率调节器，保存其状态
    if lr_scheduler is not None:
        state_dict["lr_scheduler_state"] = lr_scheduler.state_dict()
    
    # 使用 safetensors 保存（如果模型支持）
    try:
        model.save_pretrained(
            save_path,
            safe_serialization=True,
            state_dict=state_dict
        )
    except Exception as e:
        # 如果 safetensors 保存失败，使用 PyTorch 方式保存
        torch.save(state_dict, f"{save_path}.pt")
    
    logging.info(f"Success checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}")
    return



def load_training_checkpoint(model, load_dir, optimizer=None, lr_scheduler=None, **kwargs):
    """加载模型检查点"""
    if not os.path.exists(load_dir):
        logging.info(f"检查点目录 {load_dir} 不存在，从步骤 0 开始")
        return model, 0, 0
    
    # 优先选择 safetensors 格式
    safetensors_files = [f for f in os.listdir(load_dir) if f.startswith('checkpoint-') and f.endswith('.safetensors')]
    if safetensors_files:
        checkpoint_files = safetensors_files
        use_safetensors = True
    else:
        # 如果没有 safetensors 文件，使用 pt 格式
        pt_files = [f for f in os.listdir(load_dir) if f.startswith('checkpoint-') and f.endswith('.pt')]
        checkpoint_files = pt_files
        use_safetensors = False
    
    if not checkpoint_files:
        logging.info(f"在 {load_dir} 中未找到检查点，从步骤 0 开始")
        return model, 0, 0
    
    # 提取步数时考虑不同格式的后缀
    steps = [int(f.split('-')[1].replace('.safetensors', '').replace('.pt', '')) for f in checkpoint_files]
    latest_step = max(steps)
    latest_file = [f for f in checkpoint_files if str(latest_step) in f][0]
    latest_checkpoint = os.path.join(load_dir, latest_file)
    
    logging.info(f"从步骤 {latest_step} 加载检查点")
    
    try:
        # 根据格式选择加载方式
        if use_safetensors:
            from safetensors.torch import load_file
            checkpoint = load_file(latest_checkpoint)
        else:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        
        # 加载模型权重
        model.load_state_dict(checkpoint["module"])
        
        # 加载优化器状态（如果有）
        if optimizer is not None and "optimizer_state" in checkpoint:
            logging.info(f"Current optimizer type: {type(optimizer)}")
            
            try:
                # 检查是否是 DeepSpeed 优化器
                if isinstance(optimizer, accelerate.utils.deepspeed.DeepSpeedOptimizerWrapper):
                    logging.info("Detected DeepSpeed optimizer")
                    checkpoint_state = checkpoint["optimizer_state"]
                    
                    if "ds_state" in checkpoint_state:
                        # 加载完整的 DeepSpeed 状态
                        try:
                            optimizer.optimizer.load_state_dict(checkpoint_state["ds_state"])
                            logging.info("Successfully loaded complete DeepSpeed state")
                            return model, epoch, last_global_step
                        except Exception as e:
                            logging.error(f"Failed to load complete DeepSpeed state: {str(e)}")
                    
                    if "base_state" in checkpoint_state:
                        # 尝试只加载基础优化器状态
                        try:
                            optimizer.optimizer.base_optimizer.load_state_dict(checkpoint_state["base_state"])
                            logging.info("Successfully loaded base optimizer state")
                            return model, epoch, last_global_step
                        except Exception as e:
                            logging.error(f"Failed to load base optimizer state: {str(e)}")
            
                    logging.warning("Could not load any optimizer state, reinitializing")
            except Exception as e:
                logging.error(f"Error processing optimizer: {e}")
        
        # 加载学习率调节器状态（如果有）
        if lr_scheduler is not None and "lr_scheduler_state" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
                logging.info("Successfully loaded lr_scheduler state")
            except Exception as e:
                logging.warning(f"Failed to load lr_scheduler state: {e}")
        
        epoch = checkpoint["epoch"]
        last_global_step = checkpoint["last_global_step"]
        
        return model, epoch, last_global_step
        
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise e