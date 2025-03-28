python3  stage2_batchtest_inpaint_model.py \
  --img_weigh 384 \
  --img_height 512 \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --image_encoder_g_path='{image_encoder_path}' \
 --image_encoder_p_path='facebook/dinov2-giant' \
 --img_path='{image_path}' \
 --json_path='{data.json}' \
 --pose_path="{pose_path}" \
 --target_embed_path="./logs/view_stage1/512_512/" \
 --save_path="./logs/view_stage2/512_512" \
 --weights_name="{save_ckpt}" \
 --calculate_metrics
