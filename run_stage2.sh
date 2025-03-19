
accelerate launch --gpu_ids 0 --num_processes 1 --use_deepspeed --mixed_precision="bf16"  stage2_train_inpaint_model.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --image_root_path="/home/y/yuansui/tryon_stage1/data/VTON/train"  \
  --img_height=512  \
  --img_width=512   \
  --learning_rate=1e-4 \
  --train_batch_size=1 \
  --resume_from_checkpoint="logs/stage2" \
  --max_train_steps=1000000 \
  --mixed_precision="bf16" \
  --checkpointing_steps=50  \
  --noise_offset=0.1 \
  --lr_warmup_steps 5000  \
  --seed 42

#   --json_path='data/train_data.json' \