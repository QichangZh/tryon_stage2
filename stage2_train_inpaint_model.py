import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.controlnet import ControlNetConditioningEmbedding
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate  # 添加这个导入到文件顶部
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from src.configs.stage2_config import args
from test_tools import validate_and_evaluate
from datetime import datetime
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from src.dataset.stage2_dataset import InpaintDataset, InpaintCollate_fn
from transformers import CLIPVisionModelWithProjection
from transformers import Dinov2Model
from src.models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)

class ImageProjModel_p(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        return self.net(x)

class ImageProjModel_g(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # b, 257,1280
        return self.net(x)


class SDModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, unet) -> None:
        super().__init__()
        self.image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024)

        self.unet = unet
        # self.pose_proj = ControlNetConditioningEmbedding(
        #     conditioning_embedding_channels=320,
        #     block_out_channels=(16, 32, 96, 256),
        #     conditioning_channels=3)


    def forward(self, noisy_latents, timesteps, simg_f_p, timg_f_g):

        extra_image_embeddings_p = self.image_proj_model_p(simg_f_p)
        extra_image_embeddings_g = timg_f_g

        encoder_image_hidden_states = torch.cat([extra_image_embeddings_p ,extra_image_embeddings_g], dim=1)
        # pose_cond = self.pose_proj(pose_f)

        pred_noise = self.unet(noisy_latents, timesteps, class_labels=timg_f_g, encoder_hidden_states=encoder_image_hidden_states).sample
        return pred_noise




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



def main():
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)



    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load model
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path)
    image_encoder_g = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_g_path)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    unet = Stage2_InapintUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                   in_channels=9, class_embed_type="projection" ,projection_class_embeddings_input_dim=1024,
                                                  low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    image_encoder_p.requires_grad_(False)
    image_encoder_g.requires_grad_(False)
    vae.requires_grad_(False)

    sd_model = SDModel(unet=unet)

    sd_model = sd_model.to(dtype=weight_dtype)
    vae = vae.to(dtype=weight_dtype)
    image_encoder_p = image_encoder_p.to(dtype=weight_dtype)
    image_encoder_g = image_encoder_g.to(dtype=weight_dtype)
    
    sd_model.train()


    if args.gradient_checkpointing:
        sd_model.enable_gradient_checkpointing()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.train_batch_size
                * accelerator.num_processes
        )

    # Optimizer creation
    params_to_optimize = sd_model.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = InpaintDataset(
        args.image_root_path,
        size=(args.img_width, args.img_height), # w h
        imgp_drop_rate=0.1,
        imgg_drop_rate=0.1,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        collate_fn=InpaintCollate_fn,
        batch_size=args.train_batch_size,
        num_workers=2,)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 创建验证数据集
    val_dataset = InpaintDataset(
        args.val_image_root_path,  # 需要在args中添加验证集路径参数
        size=(args.img_width, args.img_height),
        imgp_drop_rate=0.0,  # 验证时不需要dropout
        imgg_drop_rate=0.0
    )

    # 验证集sampler
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, 
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False  # 验证时不需要shuffle
    )

    # 创建验证数据加载器
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=InpaintCollate_fn,
        batch_size=args.val_batch_size,  # 需要在args中添加验证batch size参数
        num_workers=2,
    )

    # 准备验证dataloader
    val_dataloader = accelerator.prepare(val_dataloader)

    # Prepare everything with our `accelerator`.
    sd_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(sd_model, optimizer, train_dataloader, lr_scheduler)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print("----------------------weight_dtype--------------------------")
    print(weight_dtype)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder_p.to(accelerator.device, dtype=weight_dtype)
    image_encoder_g.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)




    # Train!
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        sd_model, last_epoch, last_global_step = load_training_checkpoint(
            sd_model,
            args.resume_from_checkpoint,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}, global step: {last_global_step}")
        starting_epoch = last_epoch
        global_steps = last_global_step
    else:
        global_steps = 0
        starting_epoch = 0

    progress_bar = tqdm(range(global_steps, args.max_train_steps), initial=global_steps, desc="Steps",
                        # Only show the progress bar once on each machine.
                        disable=not accelerator.is_local_main_process, )

    bsz = args.train_batch_size


    for epoch in range(starting_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sd_model):
                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(batch["source_target_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Get the masked image latents
                    masked_latents = vae.encode(batch["vae_source_mask_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor

                    # mask
                    mask1 = torch.ones((bsz, 1, int(args.img_height / 8), int(args.img_width / 8))).to(accelerator.device, dtype=weight_dtype)
                    mask0 = torch.zeros((bsz, 1, int(args.img_height / 8), int(args.img_width / 8))).to(accelerator.device, dtype=weight_dtype)
                    mask = torch.cat([mask1, mask0], dim=3)
                    # Get the image embedding for conditioning
                    cond_image_feature_p = image_encoder_p(batch["cloth_image"].to(accelerator.device, dtype=weight_dtype))
                    cond_image_feature_p = (cond_image_feature_p.last_hidden_state)


                    cond_image_feature_g = image_encoder_g(batch["warp_image"].to(accelerator.device, dtype=weight_dtype), ).image_embeds
                    cond_image_feature_g =cond_image_feature_g.unsqueeze(1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.train_batch_size,),device=latents.device, )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noisy_latents = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                # Get the text embedding for conditioning


                # cond_pose = batch["source_target_pose"].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = sd_model(noisy_latents, timesteps, cond_image_feature_p,cond_image_feature_g, )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = sd_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_steps += 1

                if global_steps % args.checkpointing_steps == 0:
                    checkpoint_model(
                        args.output_dir, global_steps, sd_model, epoch, global_steps, optimizer=optimizer,lr_scheduler=lr_scheduler
                    )

                if global_steps % 50 == 0:  # 每50步验证一次
                    # 确保只在主进程进行验证和记录
                    if accelerator.is_main_process:
                        logger.info(f"Starting validation at step {global_steps}...")
                        torch.cuda.empty_cache()
                        val_loss, metrics = validate_and_evaluate(
                            sd_model, 
                            val_dataloader, 
                            vae,
                            accelerator, 
                            global_steps,
                            weight_dtype,
                            image_encoder_p,
                            image_encoder_g
                        )
                        logs.update({
                            "val_loss": val_loss,
                            "LPIPS": metrics["lpips"],
                            "SSIM": metrics["ssim"],
                            "FID": metrics["fid"],
                            "KID": metrics["kid"]
                        })
                        logger.info(f"Step {global_steps}: Validation Loss: {val_loss}")

                        # 在这里添加保存metrics的代码
                        metrics_log_dir = os.path.join(args.output_dir, 'metrics_log')
                        os.makedirs(metrics_log_dir, exist_ok=True)
                        
                        txt_path = os.path.join(metrics_log_dir, 'evaluation_metrics.txt')
                        
                        with open(txt_path, 'a') as f:
                            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                                    f"step-{global_steps} "
                                    f"lpips-{metrics['lpips']:.6f} "
                                    f"ssim-{metrics['ssim']:.6f} "
                                    f"fid-{metrics['fid']:.6f} "
                                    f"kid-{metrics['kid']:.6f}\n")

                        logger.info(f"Metrics saved to {txt_path}")
                    
                    # 等待所有进程完成验证
                    accelerator.wait_for_everyone()


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_steps >= args.max_train_steps:
                break

    # Create the pipeline using  the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":

    main()