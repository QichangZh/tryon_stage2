import logging
import os
from typing import Iterable, Optional
from packaging import version
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import  DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version
from src.dataset.stage1_dataset import PriorCollate_fn, PriorImageDataset
from src.configs.stage1_config import args
from src.models.stage1_pure_prior_transformer import Stage1_PriorTransformerPure
logger = get_logger(__name__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")


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


def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict= torch.load(load_dir, map_location="cpu")

    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    # TODO optimizer lr, and loss state

    weight_dict = checkpoint_state_dict["module"]
    new_weight_dict = {f"module.{key}": value for key, value in weight_dict.items()}
    model.load_state_dict(new_weight_dict)
    del checkpoint_state_dict

    return model, epoch, last_global_step


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6

def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).eval()
    prior = Stage1_PriorTransformerPure(
        embedding_dim=image_encoder.config.projection_dim,
        num_attention_heads=32,
        attention_head_dim=64,
        num_layers=20,
        dropout=0.1,
    )

    image_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        prior.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        prior.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.max_train_steps)

    # ------------------------------------------------------------------ data
    dataset = PriorImageDataset(
        image_root_path=args.img_path,
        size=(args.img_width, args.img_height),
        s_img_drop_rate=0.1,
        s_pose_drop_rate=0.1,
        t_pose_drop_rate=0.1,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, collate_fn=PriorCollate_fn, batch_size=args.train_batch_size, num_workers=8, pin_memory=True)

    prior, optimizer, lr_scheduler = accelerator.prepare(prior, optimizer, lr_scheduler)

    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)
    image_encoder.to(accelerator.device, dtype=dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("pure_transformer_prior", config=vars(args))

    logger.info(f"Model params — Prior: {count_params(prior):.2f}M  ·  ImageEncoder: {count_params(image_encoder):.2f}M")
    logger.info("***** Training *****")

    global_steps = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        prior.train()
        for batch in loader:
            with torch.no_grad():
                cloth = image_encoder(batch["clip_cloth_img"].to(accelerator.device, dtype=dtype)).image_embeds  # (bs,E)
                agn   = image_encoder(batch["clip_agnostic_img"].to(accelerator.device, dtype=dtype)).image_embeds.unsqueeze(1)  # (bs,1,E)
                tgt   = image_encoder(batch["clip_image_img"].to(accelerator.device, dtype=dtype)).image_embeds                 # (bs,E)

            with accelerator.accumulate(prior):
                pred = prior(proj_embedding=cloth, encoder_hidden_states=agn).predicted_image_embedding
                loss = F.mse_loss(pred.float(), tgt.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1); global_steps += 1
                accelerator.log({"train_loss": loss.item()}, step=global_steps)
                if global_steps >= args.max_train_steps:
                    break
        if global_steps >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        prior.save_pretrained(os.path.join(args.output_dir, "prior_pure"))
    accelerator.end_training()


if __name__ == "__main__":
    main()