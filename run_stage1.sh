accelerate launch --num_machines 1 --mixed_precision no --dynamo_backend no --gpu_ids 0 --use_deepspeed --num_processes 1   \
  stage1_train_prior_model.py \
  --pretrained_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
  --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --img_path="./data/test" \
  --output_dir="output_dir" \
  --img_height=512  \
  --img_width=512   \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000 \
  --noise_offset=0.1 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --checkpointing_steps=5000 \
  --seed 42

  # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
  # laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k
  # openai/clip-vit-large-patch14
  # --json_path="./data/train_data.json" \