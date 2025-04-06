import os
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler
from src.models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel
from src.pipelines.stage2_inpaint_pipeline import Stage2_InpaintDiffusionPipeline
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
import argparse
from transformers import Dinov2Model
from typing import Any, Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as compare_ssim
import glob
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import json
import time


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


class InpaintDataset(Dataset):
    """虚拟试衣数据集"""
    
    def __init__(self, img_path, img_width, img_height, transform=None):
        """
        初始化数据集
        
        Args:
            img_path: 图像根目录
            img_width: 图像宽度
            img_height: 图像高度
            transform: 转换函数
        """
        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        
        # 获取所有图像文件名
        self.image_files = self._get_image_files()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
        )
        
    def _get_image_files(self):
        """获取目录中的所有图像文件"""
        image_dir = os.path.join(self.img_path, "image")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        print(f"找到 {len(image_files)} 个图像文件")
        return [os.path.basename(f) for f in image_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 构建各类图像路径
        cloth_img_path = os.path.join(self.img_path, "cloth", img_name)
        warp_img_path = os.path.join(self.img_path, "warp_mask", img_name)
        t_img_path = os.path.join(self.img_path, "image", img_name)
        mask_img_path = os.path.join(self.img_path, "mask", img_name)
        black_img_path = os.path.join(self.img_path, "black_cloth", img_name)
        
        # 读取并调整图像大小
        cloth_img = Image.open(cloth_img_path).convert("RGB").resize((self.img_width, self.img_height), Image.BICUBIC)
        warp_img = Image.open(warp_img_path).convert("RGB").resize((self.img_width, self.img_height), Image.BICUBIC)
        t_img = Image.open(t_img_path).convert("RGB").resize((self.img_width, self.img_height), Image.BICUBIC)
        mask_img = Image.open(mask_img_path).convert("RGB").resize((self.img_width, self.img_height), Image.BICUBIC)
        black_img = Image.open(black_img_path).convert("RGB").resize((self.img_width, self.img_height), Image.BICUBIC)
        
        # 创建遮罩组合图像
        s_img_t_mask = Image.new("RGB", (self.img_width * 2, self.img_height))
        s_img_t_mask.paste(cloth_img, (0, 0))
        s_img_t_mask.paste(black_img, (self.img_width, 0))

        to_tensor = transforms.ToTensor()
        
        cloth_img_tensor = to_tensor(cloth_img)
        warp_img_tensor = to_tensor(warp_img)
        t_img_tensor = self.transform(t_img)
        mask_img_tensor = self.transform(mask_img)
        black_img_tensor = self.transform(black_img)
        s_img_t_mask_tensor = self.transform(s_img_t_mask)
        
        return {
            'cloth_img_tensor': cloth_img_tensor,
            'warp_img_tensor': warp_img_tensor,
            't_img_tensor': t_img_tensor,
            'mask_img_tensor': mask_img_tensor,
            'black_img_tensor': black_img_tensor,
            's_img_t_mask_tensor': s_img_t_mask_tensor,
        }
    

def InpaintCollate_fn(data):
    """
    将数据集中的样本整理成批次
    
    Args:
        data: 一个包含多个样本的列表，每个样本是由 __getitem__ 方法返回的字典
        
    Returns:
        一个包含批次数据的字典
    """
    cloth_img_tensor = torch.stack([example["cloth_img_tensor"] for example in data])
    cloth_img_tensor = cloth_img_tensor.to(memory_format=torch.contiguous_format).float()
    
    warp_img_tensor = torch.stack([example["warp_img_tensor"] for example in data])
    warp_img_tensor = warp_img_tensor.to(memory_format=torch.contiguous_format).float()
    
    t_img_tensor = torch.stack([example["t_img_tensor"] for example in data])
    t_img_tensor = t_img_tensor.to(memory_format=torch.contiguous_format).float()
    
    mask_img_tensor = torch.stack([example["mask_img_tensor"] for example in data])
    mask_img_tensor = mask_img_tensor.to(memory_format=torch.contiguous_format).float()
    
    black_img_tensor = torch.stack([example["black_img_tensor"] for example in data])
    black_img_tensor = black_img_tensor.to(memory_format=torch.contiguous_format).float()
    
    s_img_t_mask_tensor = torch.stack([example["s_img_t_mask_tensor"] for example in data])
    s_img_t_mask_tensor = s_img_t_mask_tensor.to(memory_format=torch.contiguous_format).float()
    
    return {
        "cloth_img": cloth_img_tensor,
        "warp_img": warp_img_tensor,
        "t_img": t_img_tensor,
        "mask_img": mask_img_tensor,
        "black_img": black_img_tensor,
        "s_img_t_mask": s_img_t_mask_tensor,
    }
    

def inference(args):
    # 设置设备
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # 保存路径
    save_dir = "{}/show_guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)
    save_dir_metric = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    # 图像处理器和变换
    clip_image_processor = CLIPImageProcessor()
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 定义模型
    image_proj_model_p_dict = {}
    pose_proj_dict = {}
    unet_dict = {}

    # 加载预训练模型
    image_encoder_g = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_g_path).to(device).eval()
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path).to(device).eval()
    image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).eval()

    # 从Hugging Face Hub下载检查点
    repo_id = "bailuyucha/stage2"
    filename = f"{args.step_number}/mp_rank_00_model_states.pt"
    from huggingface_hub import hf_hub_download
    print(f"正在从 {repo_id} 下载检查点，文件：{filename}")
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"检查点下载到 {weights_path}")

    # 加载模型权重
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

    # 加载Image Projection模型权重
    image_proj_model_p.load_state_dict(image_proj_model_p_dict)

    # 设置Pipeline
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

    # 初始化VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    
    # 创建数据集和DataLoader
    dataset = InpaintDataset(
        img_path=args.img_path,
        img_width=args.img_width,
        img_height=args.img_height,
        transform=None  # 数据集内部已有转换
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,  # 推理时通常batch_size=1
        shuffle=False,
        num_workers=4,
        collate_fn=InpaintCollate_fn,
    )
    
    # 指标列表
    all_ssim = []
    start_time = time.time()
    
    # 遍历数据集进行推理
    batch_counter = 0
    for batch in dataloader:
        cloth_img = batch['cloth_img'].to(device)
        t_img = batch['t_img'].to(device)
        warp_img = batch['warp_img'].to(device)
        mask_img = batch['mask_img'].to(device)
        s_img_t_mask = batch['s_img_t_mask'].to(device)

        # 处理掩码
        # mask_tensor = mask_img.unsqueeze(0)
        mask0 = vae.encode(mask_img).latent_dist.sample()
        mask0 = mask0 * vae.config.scaling_factor
        mask0 = mask0[:, :1, :, :].to(dtype=torch.float32)  # 只保留第一个通道

        # 创建完整掩码
        batch_size = mask_img.shape[0] 
        mask1 = torch.ones((batch_size, 1, int(args.img_height / 8), int(args.img_width / 8))).to(device, dtype=torch.float32)
        mask = torch.cat([mask1, mask0], dim=3)

        # 处理服装图像特征
        clip_processor_cloth_img = clip_image_processor(images=cloth_img, return_tensors="pt").pixel_values
        cloth_img_f = image_encoder_p(clip_processor_cloth_img.to(device)).last_hidden_state
        cloth_img_proj_f = image_proj_model_p(cloth_img_f)


        clip_processor_s_img = clip_image_processor(images=warp_img, return_tensors="pt").pixel_values
        pred_t_img_embed = (image_encoder_g(clip_processor_s_img.to(device)).image_embeds).unsqueeze(1)

        # 准备VAE输入
        vae_image = s_img_t_mask

        # 进行推理
        output = pipe(
            height=args.img_height,
            width=args.img_width*2,
            guidance_rescale=0.0,
            vae_image=vae_image,
            s_img_proj_f=cloth_img_proj_f,
            pred_t_img_embed=pred_t_img_embed,
            num_images_per_prompt=4,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            mask=mask,
        )
        for i in range(batch_size):

            # 创建可视化图像
            vis_st_image = Image.new("RGB", (args.img_width*2, args.img_height))

            # 反归一化处理
            # 如果图像在[-1, 1]范围内，需要转换回[0, 1]范围
            cloth_img_denorm = cloth_img[i].cpu().clone()
            t_img_denorm = t_img[i].cpu().clone()

            # 反归一化: x = x * 0.5 + 0.5
            # cloth_img_denorm = cloth_img_denorm * 0.5 + 0.5
            t_img_denorm = t_img_denorm * 0.5 + 0.5

            # 将PyTorch张量转换为PIL图像
            cloth_img_pil = to_pil_image(cloth_img_denorm)
            t_img_pil = to_pil_image(t_img_denorm)

            vis_st_image.paste(cloth_img_pil, (0, 0))
            vis_st_image.paste(t_img_pil, (args.img_width, 0))

            # 计算指标或保存结果
            if args.calculate_metrics:
                ssim_values = []
                for gen_img in output.images:
                    gen_img = gen_img.crop((args.img_width, 0, args.img_width*2, args.img_height))
                    ssim_values.append(compare_ssim(
                        np.array(t_img_pil)*255.0, 
                        np.array(gen_img)*255.0,
                        gaussian_weights=True, 
                        sigma=1.2,
                        use_sample_covariance=False, 
                        multichannel=True, 
                        channel_axis=2,
                        data_range=(np.array(gen_img)*255.0).max() - (np.array(gen_img)*255.0).min()
                    ))
                    # 修改后的代码
                    max_value = max(ssim_values)
                    all_ssim.append(max_value)
                    max_index = ssim_values.index(max_value)
                    grid_metric = output.images[max_index].crop((args.img_width, 0, args.img_width*2, args.img_height))
                    img_name = f"batch_{batch_counter:04d}_sample_{i:02d}_ssim_{max_value:.4f}_metric.png"
                    grid_metric.save(os.path.join(save_dir_metric, img_name))
            else:
                sample_images = [output.images[i]] if hasattr(output, 'images') else []
                sample_images.insert(0, vis_st_image)
                grid = image_grid(sample_images, 1, len(sample_images))
                img_name = f"batch_{batch_counter:04d}_sample_{i:02d}_metric.png"
                grid.save(os.path.join(save_dir, img_name))

        batch_counter += 1

    # 计算总时间和平均指标
    end_time = time.time()
    print(f"总推理时间: {end_time-start_time:.2f}秒")

    if args.calculate_metrics and all_ssim:
        avg_ssim = sum(all_ssim) / len(all_ssim)
        print(f"平均SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用DataLoader的虚拟试衣推理脚本")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-2-1-base",
                        help="预训练模型路径或huggingface.co/models中的模型标识符")
    parser.add_argument("--image_encoder_g_path", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        help="预训练模型路径或huggingface.co/models中的模型标识符")
    parser.add_argument("--image_encoder_p_path", type=str, default="facebook/dinov2-giant",
                        help="预训练模型路径或huggingface.co/models中的模型标识符")
    parser.add_argument("--img_path", type=str, default="/root/autodl-tmp/data/test", help="图像路径")
    parser.add_argument("--save_path", type=str, default="./logs/view_stage2/384_512", help="保存路径")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="引导尺度")
    parser.add_argument("--seed_number", type=int, default=42, help="随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument("--img_width", type=int, default=384, help="图像宽度")
    parser.add_argument("--img_height", type=int, default=512, help="图像高度")
    parser.add_argument("--calculate_metrics", action='store_true', help="计算SSIM指标")
    parser.add_argument("--step_number", type=str, default="30000", help="Hugging Face仓库中的模型步数（例如12000）")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小（推理时通常为1）")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader工作线程数")
    
    args = parser.parse_args()
    print(args)
    
    inference(args)
