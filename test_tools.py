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

# 添加验证和评估函数
def compute_metrics(generated_images, real_images):
    """计算各种评估指标"""
    # 初始化LPIPS模型
    loss_fn_alex = lpips.LPIPS(net='alex').to(generated_images.device)
    
    # 转换图像格式
    generated_np = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
    real_np = real_images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 计算LPIPS
    lpips_value = loss_fn_alex(generated_images, real_images).mean().item()
    
    # 计算SSIM
    ssim_value = np.mean([ssim(generated_np[i], real_np[i], multichannel=True) 
                         for i in range(len(generated_np))])
    
    # 保存临时图像用于计算FID和KID
    temp_gen_dir = "temp_generated"
    temp_real_dir = "temp_real"
    os.makedirs(temp_gen_dir, exist_ok=True)
    os.makedirs(temp_real_dir, exist_ok=True)
    
    for i in range(len(generated_images)):
        save_image(generated_images[i], f"{temp_gen_dir}/{i}.png")
        save_image(real_images[i], f"{temp_real_dir}/{i}.png")
    
    # 计算FID
    fid_value = fid.compute_fid(temp_gen_dir, temp_real_dir)
    
    # 计算KID
    metrics = torch_fidelity.calculate_metrics(
        input1=temp_gen_dir,
        input2=temp_real_dir,
        metrics=['kid'],
    )
    kid_value = metrics['kid']
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_real_dir)
    
    return lpips_value, ssim_value, fid_value, kid_value

def validate_and_evaluate(sd_model, val_dataloader, vae, accelerator, global_step, weight_dtype, image_encoder_p, image_encoder_g):
    """验证函数"""
    sd_model.eval()
    total_val_loss = 0
    all_generated_images = []
    all_real_images = []
    
    # 获取noise scheduler (应从训练代码传入)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    with torch.no_grad():
        for batch in val_dataloader:
            # 验证过程与训练过程保持一致
            latents = vae.encode(batch["source_target_image"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

            # 准备条件输入
            masked_latents = vae.encode(batch["vae_source_mask_image"].to(dtype=weight_dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            # 准备mask
            mask1 = torch.ones((batch["source_target_image"].shape[0], 1, int(args.img_height/8), int(args.img_width/8))).to(latents.device)
            mask0 = torch.zeros_like(mask1)
            mask = torch.cat([mask1, mask0], dim=3)

            # 组合输入
            noisy_latents = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            # 准备条件特征
            cond_image_feature_p = image_encoder_p(batch["cloth_image"].to(dtype=weight_dtype)).last_hidden_state
            cond_image_feature_g = image_encoder_g(batch["warp_image"].to(dtype=weight_dtype)).image_embeds.unsqueeze(1)
            
            # 生成预测
            model_pred = sd_model(noisy_latents, timesteps, cond_image_feature_p, cond_image_feature_g)
            
            # 计算损失
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            total_val_loss += loss.item()
            
            # 解码生成的图像
            generated_images = vae.decode(latents).sample
            all_generated_images.append(generated_images)
            all_real_images.append(batch["source_target_image"])
    
    # 计算平均验证损失
    avg_val_loss = total_val_loss / len(val_dataloader)
    
    # 计算评估指标
    all_generated_images = torch.cat(all_generated_images, dim=0)
    all_real_images = torch.cat(all_real_images, dim=0)
    
    lpips_value, ssim_value, fid_value, kid_value = compute_metrics(
        all_generated_images, all_real_images
    )
    
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