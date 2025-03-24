srun accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    --use_deepspeed \
    --mixed_precision="bf16" \
    stage2_train_inpaint_model.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --image_encoder_p_path='facebook/dinov2-giant' \
    --image_encoder_g_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --image_root_path="/home/y/yuansui/data/VTON/train"  \
    --img_height=512  \
    --img_width=384   \
    --learning_rate=1e-4 \
    --train_batch_size=48 \
    --val_batch_size=32 \
    --max_train_steps=100000 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="bf16" \
    --checkpointing_steps=2000  \
    --noise_offset=0.1 \
    --lr_warmup_steps 5000  \
    --seed 42

    # --resume_from_checkpoint="logs/stage2" \