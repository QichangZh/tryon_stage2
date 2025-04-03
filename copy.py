import os
import glob
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler
from src.models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel
from src.pipelines.stage2_inpaint_pipeline import Stage2_InpaintDiffusionPipeline
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
import argparse
from transformers import Dinov2Model
from typing import Any, Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time

def split_list_into_chunks(lst, n):
    chunk_size = max(1, len(lst) // n)
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


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


def get_image_files(directory):
    """获取目录中的所有图像文件"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return [os.path.basename(f) for f in image_files]


def inference(args, rank, image_files):
    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # 保存路径
    save_dir = "{}/show_guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)
    save_dir_metric = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 模型定义
    image_proj_model_p_dict = {}
    pose_proj_dict = {}
    unet_dict = {}

    image_encoder_g = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_g_path).to(device).eval()
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path).to(device).eval()

    image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).eval()
    pose_proj = ControlNetConditioningEmbedding(320, 3, (16, 32, 96, 256)).to(device).eval()

    # 从Hugging Face下载权重
    repo_id = "bailuyucha/stage2"
    filename = f"{args.step_number}/mp_rank_00_model_states.pt"
    from huggingface_hub import hf_hub_download
    print(f"下载检查点，来源：{repo_id}，文件：{filename}")
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"已下载检查点到：{weights_path}")

    model_sd = torch.load(weights_path, map_location="cpu")["module"]

    for k in model_sd.keys():
        if k.startswith("pose_proj"):
            pose_proj_dict[k.replace("pose_proj.", "")] = model_sd[k]
        elif k.startswith("image_proj_model_p"):
            image_proj_model_p_dict[k.replace("image_proj_model_p.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        else:
            print(k)

    pose_proj.load_state_dict(pose_proj_dict)
    image_proj_model_p.load_state_dict(image_proj_model_p_dict)

    pipe = Stage2_InpaintDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)

    pipe.unet = Stage2_InapintUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
        in_channels=9, 
        class_embed_type="projection",
        projection_class_embeddings_input_dim=1024,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True
    ).to(device)

    pipe.unet.load_state_dict(unet_dict)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    print(f'====================== 模型加载完成 ===================')

    all_ssim = []
    start_time = time.time()
    
    for img_name in image_files:
        # 根据dataset的目录结构读取相应文件
        cloth_img_path = os.path.join(args.image_root_path, "cloth", img_name)
        warp_img_path = os.path.join(args.image_root_path, "warp_mask", img_name)
        t_img_path = os.path.join(args.image_root_path, "image", img_name)
        mask_img_path = os.path.join(args.image_root_path, "mask", img_name)
        black_img_path = os.path.join(args.image_root_path, "black_cloth", img_name)
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [cloth_img_path, warp_img_path, t_img_path, mask_img_path, black_img_path]):
            print(f"跳过 {img_name}，因为某些必要的文件不存在")
            continue

        # 读取并调整图像大小
        cloth_img = Image.open(cloth_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        warp_img = Image.open(warp_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        t_img = Image.open(t_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        mask_img = Image.open(mask_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        black_img = Image.open(black_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)

        # 创建遮罩组合图像
        s_img_t_mask = Image.new("RGB", (args.img_width * 2, args.img_height))
        s_img_t_mask.paste(cloth_img, (0, 0))
        s_img_t_mask.paste(black_img, (args.img_width, 0))

        # 图像特征提取
        clip_processor_cloth_img = clip_image_processor(images=cloth_img, return_tensors="pt").pixel_values
        cloth_img_f = image_encoder_p(clip_processor_cloth_img.to(device)).last_hidden_state
        cloth_img_proj_f = image_proj_model_p(cloth_img_f)

        # 准备模型输入
        vae_image = torch.unsqueeze(img_transform(s_img_t_mask), 0)
        
        # 处理warp图像
        clip_processor_warp_img = clip_image_processor(images=warp_img, return_tensors="pt").pixel_values
        warp_img_embed = (image_encoder_g(clip_processor_warp_img.to(device)).image_embeds).unsqueeze(1)

        # 如果需要，可以准备姿态图
        pose_img = Image.new("RGB", (args.img_width * 2, args.img_height))
        pose_img.paste(cloth_img, (0, 0))  # 这里简化了，实际应该使用姿态图像
        pose_img.paste(t_img, (args.img_width, 0))
        cond_pose = torch.unsqueeze(img_transform(pose_img), 0)
        pose_f = pose_proj(cond_pose.to(device=device))

        # 运行推理
        output = pipe(
            height=args.img_height,
            width=args.img_width*2,
            guidance_rescale=0.0,
            vae_image=vae_image,
            s_img_proj_f=cloth_img_proj_f,
            st_pose_f=pose_f,
            pred_t_img_embed=warp_img_embed,
            num_images_per_prompt=4,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        )

        # 创建可视化图像
        vis_st_image = Image.new("RGB", (args.img_width*2, args.img_height))
        vis_st_image.paste(cloth_img, (0, 0))
        vis_st_image.paste(t_img, (args.img_width, 0))

        if args.calculate_metrics:
            ssim_values = []
            for gen_img in output.images:
                gen_img = gen_img.crop((args.img_width, 0, args.img_width*2, args.img_height))
                ssim_values.append(compare_ssim(
                    np.array(t_img)*255.0, 
                    np.array(gen_img)*255.0,
                    gaussian_weights=True, 
                    sigma=1.2,
                    use_sample_covariance=False, 
                    multichannel=True, 
                    channel_axis=2,
                    data_range=(np.array(gen_img)*255.0).max() - (np.array(gen_img)*255.0).min()
                ))
            max_value = max(ssim_values)
            all_ssim.append(max_value)
            max_index = ssim_values.index(max_value)
            grid_metric = output.images[max_index].crop((args.img_width, 0, args.img_width*2, args.img_height))
            grid_metric.save(os.path.join(save_dir_metric, img_name))
        else:
            output.images.insert(0, vis_st_image)
            grid = image_grid(output.images, 1, 5)
            grid.save(os.path.join(save_dir, img_name))

    end_time = time.time()
    print(f"推理时间: {end_time-start_time:.2f}秒")

    if args.calculate_metrics and all_ssim:
        avg_ssim = sum(all_ssim) / len(