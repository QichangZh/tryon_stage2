import logging
import math
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from src.configs.stage3_config import args

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from src.dataset.stage3_dataset import RefinedDataset,RefinedCollate_fn
from transformers import Dinov2Model



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

    def forward(self, x):  # b, 257,1280
        return self.net(x)


class SDModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, unet) -> None:
        super().__init__()
        self.image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024)

        self.unet = unet


    def forward(self, noisy_latents, timesteps,  s_img_embed):  # img_f 8,1024
        """
        encoder_hidden_states : 8,77, 1024
        control_image_feature:  8, 257, 1024
        """

        extra_image_embeddings_p = self.image_proj_model_p(s_img_embed)  # s_img: bs,257,1536

        pred_noise = self.unet(noisy_latents, timesteps, encoder_hidden_states=extra_image_embeddings_p).sample
        return pred_noise




def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict= torch.load(load_dir, map_location="cpu", weights_only=False)


    print(checkpoint_state_dict.keys())
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    # TODO optimizer lr, and loss state

    weight_dict = checkpoint_state_dict["module"]
    new_weight_dict = {f"module.{key}": value for key, value in weight_dict.items()}
    model.load_state_dict(new_weight_dict)
    del checkpoint_state_dict

    return model, epoch, last_global_step

def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return



def main():
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
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
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_path)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    
    # 2. train

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",revision=args.revision,
                                                   in_channels=8, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    # image_encoder_g.requires_grad_(False)
    image_encoder_p.requires_grad_(False)
    vae.requires_grad_(False)

    sd_model = SDModel(unet=unet)
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

    dataset = RefinedDataset(
        args.json_path,
        image_root_path=args.img_path,
        gen_t_img_path=args.gen_t_img_path,
        size=(512,512),
        s_img_drop_rate=0.1)


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        collate_fn=RefinedCollate_fn,
        batch_size=args.train_batch_size,
        num_workers=8,)

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

    # Prepare everything with our `accelerator`.
    sd_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(sd_model, optimizer, train_dataloader, lr_scheduler)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder_p.to(accelerator.device, dtype=weight_dtype)


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
        prior_model, last_epoch, last_global_step = load_training_checkpoint(
            sd_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}, global step: {last_global_step}")
        starting_epoch = last_epoch
        global_steps = last_global_step
        sd_model = sd_model
    else:
        global_steps = 0
        starting_epoch = 0
        sd_model = sd_model

    progress_bar = tqdm(range(global_steps, args.max_train_steps), initial=global_steps, desc="Steps",
                        # Only show the progress bar once on each machine.
                        disable=not accelerator.is_local_main_process, )



    for epoch in range(starting_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sd_model):
                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(batch["target_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor



                    # Get gen image setting
                    gen_latents = vae.encode(batch["gen_target_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    gen_latents = gen_latents * vae.config.scaling_factor


                    # Get the image embedding for conditioning
                    cond_image_feature_p = image_encoder_p(batch["source_image"].to(accelerator.device, dtype=weight_dtype))
                    cond_image_feature_p = (cond_image_feature_p.last_hidden_state)



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
                noisy_latents = torch.cat([noisy_latents, gen_latents], dim=1)

                # Predict the noise residual
                model_pred = sd_model(noisy_latents, timesteps, cond_image_feature_p)

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
                        args.output_dir, global_steps, sd_model, epoch, global_steps
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_steps >= args.max_train_steps:
                break

    # Create the pipeline using  the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        sd_model = accelerator.unwrap_model(sd_model)
        checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)
    accelerator.end_training()


if __name__ == "__main__":

    main()