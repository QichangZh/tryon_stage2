# 在文件开头添加必要的import
import lpips
from skimage.metrics import structural_similarity as ssim
from src.configs.stage2_config import args
import numpy as np
from cleanfid import fid
import torch_fidelity
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
import os
import shutil
def validate_and_evaluate(sd_model, val_dataloader, vae, accelerator, global_step, weight_dtype, image_encoder_p, image_encoder_g):
    """验证函数：分批计算LPIPS/SSIM并保存生成的图像，用于后续FID/KID的计算，避免一次性OOM"""
    sd_model.eval()
    total_val_loss = 0.0
    
    # 创建用于保存临时图像的目录
    temp_gen_dir = "temp_generated"
    temp_real_dir = "temp_real"
    os.makedirs(temp_gen_dir, exist_ok=True)
    os.makedirs(temp_real_dir, exist_ok=True)

    # 初始化 LPIPS
    lpips_model = lpips.LPIPS(net='alex').cuda()

    # 用于累加 LPIPS 和 SSIM
    lpips_accum = 0.0
    ssim_accum = 0.0
    count_samples = 0

    # 获取noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # ===================== 1) 编码并加噪 =====================
            real_images = batch["source_target_image"].to(dtype=weight_dtype)
            latents = vae.encode(real_images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 准备 masked_latents
            masked_latents = vae.encode(
                batch["vae_source_mask_image"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            # 准备 mask
            mask1 = torch.ones(
                (real_images.shape[0], 1, int(args.img_height/8), int(args.img_width/8)),
                device=latents.device
            )
            mask0 = torch.zeros_like(mask1)
            mask = torch.cat([mask1, mask0], dim=3)

            # 组合输入
            noisy_latents = torch.cat([noisy_latents, mask, masked_latents], dim=1).to(dtype=weight_dtype)

            # ===================== 2) 准备条件特征 =====================
            cond_image_feature_p = image_encoder_p(batch["cloth_image"].to(dtype=weight_dtype)).last_hidden_state
            cond_image_feature_g = image_encoder_g(batch["warp_image"].to(dtype=weight_dtype)).image_embeds.unsqueeze(1)

            # ===================== 3) 预测 & 计算Loss =====================
            model_pred = sd_model(noisy_latents, timesteps, cond_image_feature_p, cond_image_feature_g)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            total_val_loss += loss.item()

            # ===================== 4) 解码生成 & 分批计算LPIPS/SSIM =====================
            generated_images = vae.decode(latents).sample

            # 确保数据类型正确 - 先转换为float32
            generated_images = generated_images.to(dtype=torch.float32)
            real_images_float = real_images.to(dtype=torch.float32)

            # LPIPS计算
            lpips_batch = lpips_model(generated_images, real_images_float).mean().item()

            # SSIM计算
            gen_np = generated_images.detach().cpu().numpy().transpose(0,2,3,1)
            real_np = real_images_float.detach().cpu().numpy().transpose(0,2,3,1)
            
            ssim_batch = 0.0
            for i in range(gen_np.shape[0]):
                # 确保图像在0-1范围内
                gen_img = np.clip(gen_np[i], 0, 1)
                real_img = np.clip(real_np[i], 0, 1)
                
                # 设置较小的window_size，并明确指定channel_axis
                min_side = min(gen_img.shape[0], gen_img.shape[1])
                win_size = min(3, min_side) # 使用3或更小的window size
                if win_size % 2 == 0:  # 确保win_size是奇数
                    win_size -= 1
                
                if win_size >= 3:  # 只在window size至少为3时计算SSIM
                    ssim_val = ssim(gen_img, real_img, 
                                  win_size=win_size,
                                  channel_axis=2,  # 指定颜色通道轴
                                  data_range=1.0)  # 指定数据范围为0-1
                    ssim_batch += ssim_val
                else:
                    print(f"Warning: Image too small for SSIM calculation. Shape: {gen_img.shape}")
                    ssim_batch += 0  # 或者其他替代值

            ssim_batch /= gen_np.shape[0]

            batch_size = real_images.size(0)
            lpips_accum += lpips_batch * batch_size
            ssim_accum += ssim_batch * batch_size
            count_samples += batch_size

            # ===================== 5) 保存图像用于FID/KID =====================
            for i in range(batch_size):
                idx_global = batch_idx * val_dataloader.batch_size + i
                save_image(generated_images[i], f"{temp_gen_dir}/{idx_global}.png")
                save_image(real_images_float[i], f"{temp_real_dir}/{idx_global}.png")

            # 清理显存
            torch.cuda.empty_cache()

    # ===================== 整个验证集循环结束 =====================
    avg_val_loss = total_val_loss / len(val_dataloader)
    lpips_value = lpips_accum / count_samples
    ssim_value = ssim_accum / count_samples

    # 计算 FID
    fid_value = fid.compute_fid(temp_gen_dir, temp_real_dir)

    # 计算 KID
    metrics_kid = torch_fidelity.calculate_metrics(
        input1=temp_gen_dir,
        input2=temp_real_dir,
        metrics=['kid'],
    )
    kid_value = metrics_kid['kid']

    # 清理临时文件夹
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_real_dir)

    # 使用accelerator记录指标
    if accelerator.is_main_process:
        accelerator.log({
            "Validation/Loss": avg_val_loss,
            "Metrics/LPIPS": lpips_value,
            "Metrics/SSIM": ssim_value,
            "Metrics/FID": fid_value,
            "Metrics/KID": kid_value
        }, step=global_step)

    metrics = {
        "lpips": lpips_value,
        "ssim": ssim_value,
        "fid": fid_value,
        "kid": kid_value
    }

    sd_model.train()
    return avg_val_loss, metrics
