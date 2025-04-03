import os
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
import json
import time

def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
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
    image_dir = os.path.join(directory, "image")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    print(len(image_files))
    return [os.path.basename(f) for f in image_files]



def inference(args, rank, select_test_datas):

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)


    # save path
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


    # model define
    image_proj_model_p_dict = {}
    pose_proj_dict = {}
    unet_dict = {}

    image_encoder_g = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_g_path).to(device).eval()
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path).to(device).eval()

    image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).eval()
    # pose_proj = ControlNetConditioningEmbedding(320, 3, (16, 32, 96, 256)).to(device).eval()

    repo_id = "bailuyucha/stage2"
    filename = f"{args.step_number}/mp_rank_00_model_states.pt"
    from huggingface_hub import hf_hub_download
    print(f"Downloading checkpoint from {repo_id}, file: {filename}")
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded checkpoint to {weights_path}")

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

    # pose_proj.load_state_dict(pose_proj_dict)
    image_proj_model_p.load_state_dict(image_proj_model_p_dict)

    pipe = Stage2_InpaintDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,torch_dtype=torch.float16).to(device)

    pipe.unet= Stage2_InapintUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                           in_channels=9, class_embed_type="projection",
                                           projection_class_embeddings_input_dim=1024,torch_dtype=torch.float16,
                                           low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

    pipe.unet.load_state_dict(unet_dict)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    print(f'====================== 模型加载完成 ===================')


    # number = 0
    all_ssim = []

    start_time = time.time()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)


    for img_name in image_files:

        cloth_img_path = os.path.join(args.img_path, "cloth", img_name)
        warp_img_path = os.path.join(args.img_path, "warp_mask", img_name)
        t_img_path = os.path.join(args.img_path, "image", img_name)
        mask_img_path = os.path.join(args.img_path, "mask", img_name)
        black_img_path = os.path.join(args.img_path, "black_cloth", img_name)
        
        # 检查文件是否存在
        # if not all(os.path.exists(p) for p in [cloth_img_path, warp_img_path, t_img_path, mask_img_path, black_img_path]):
        #     print(f"跳过 {img_name}，因为某些必要的文件不存在")
        #     continue

        # 读取并调整图像大小
        cloth_img = Image.open(cloth_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        warp_img = Image.open(warp_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        t_img = Image.open(t_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        mask_img = Image.open(mask_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        black_img = Image.open(black_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)

        mask0 = img_transform(mask_img).to(dtype=torch.float32).unsqueeze(0)
        # mask0 = vae.encode(batch["vae_mask_img"].to(dtype=weight_dtype)).latent_dist.sample()
        mask0 = vae.encode(mask0).latent_dist.sample()
        mask0 = mask0 * vae.config.scaling_factor
        mask0 = mask0[:, :1, :, :]  # 只保留第一个通道，形状变为[B, 1, h, w]

        # mask
        mask1 = torch.ones((1, 1, int(args.img_height / 8), int(args.img_width / 8))).to(dtype=torch.float32)
        # mask0 = torch.zeros((bsz, 1, int(args.img_height / 8), int(args.img_width / 8))).to(accelerator.device, dtype=weight_dtype)
        mask = torch.cat([mask1, mask0], dim=3)


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

        output = pipe(
                height=args.img_height,
                width=args.img_width*2,
                guidance_rescale=0.0,
                vae_image=vae_image,
                s_img_proj_f=cloth_img_proj_f,
                pred_t_img_embed = warp_img_embed,
                num_images_per_prompt=4,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
                mask=mask,
            )


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

    end_time =time.time()
    print(end_time-start_time)

    if args.calculate_metrics:
        print(sum(all_ssim)/ len(all_ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of an inpaint model of stage2 script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-2-1-base",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument("--image_encoder_g_path",type=str,default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--image_encoder_p_path",type=str,default="facebook/dinov2-giant",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--img_path", type=str,default="/root/autodl-tmp/data/test", help="image path", )
    parser.add_argument("--pose_path", type=str,default="./datasets/deepfashing/openpose_all_img/",help="pose path", )
    # parser.add_argument("--json_path", type=str,default="./datasets/deepfashing/test_data.json",help="json path", )
    parser.add_argument("--target_embed_path", type=str,default="./logs/view_stage1/384_512/",help="t_img_embed path", )
    parser.add_argument("--save_path", type=str, default="./logs/view_stage2/384_512", help="save path", )
    parser.add_argument("--guidance_scale",type=int,default=2.0,help="guidance_scale",)
    parser.add_argument("--seed_number",type=int,default=42,help="seed number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="num_inference_steps",)
    parser.add_argument("--img_width",type=int,default=384,help="image width",)
    parser.add_argument("--img_height",type=int,default=512,help="image height",)
    parser.add_argument("--calculate_metrics",  action='store_true', help="caculate ssim", )
    # parser.add_argument("--weights_name", type=str, default="./Checkpoints/stage2_checkpoints/512",help="weights number", )
    
    
    parser.add_argument("--step_number", type=str, default="12000", help="Step number of the model in Hugging Face repo (e.g., 12000)")
    
    args = parser.parse_args()
    print(args)

    num_devices = torch.cuda.device_count()
    print("using {} num_processes inference".format(num_devices))

    image_files = get_image_files(args.img_path)

    inference(args, 0, image_files)


    # processes = []
    # for rank in range(num_devices):
    #     p = mp.Process(target=inference, args=(args, rank, data_list[rank] ))
    #     processes.append(p)
    #     p.start()

    # for rank, p in enumerate(processes):
    #     p.join()



