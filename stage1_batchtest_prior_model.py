import os
import torch
from PIL import Image
from src.models.stage1_prior_transformer import Stage1_PriorTransformer
from src.pipelines.stage1_prior_pipeline import Stage1_PriorPipeline
import torch.nn.functional as F
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from torch.utils.data import DataLoader
from src.dataset.stage1_dataset import PriorImageDataset, PriorCollate_fn
import glob
import argparse
import numpy as np
import torch.multiprocessing as mp
import json
import time
from tqdm import tqdm


def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks


class PriorTestDataset(PriorImageDataset):
    """继承自PriorImageDataset的测试数据集类"""
    
    def __init__(self, select_test_datas, size=(512, 512), image_root_path=""):
        super().__init__(size=size, image_root_path=image_root_path)
        # 重写image_files为指定的测试列表
        self.image_files = [os.path.basename(f) for f in select_test_datas]
        print(f"加载了 {len(self.image_files)} 个测试图像文件")
        
    def __getitem__(self, idx):
        # 获取原始数据
        result = super().__getitem__(idx)
        # 添加图像名称用于保存结果
        result["img_name"] = self.image_files[idx]
        return result


def main(args, rank, select_test_datas):
    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # 保存路径
    save_dir = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 创建测试数据集和DataLoader
    test_dataset = PriorTestDataset(
        select_test_datas=select_test_datas,
        size=(args.img_width, args.img_height),
        image_root_path=args.img_path
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PriorCollate_fn,
        pin_memory=True
    )

    # 加载模型
    model_ckpt = args.weights_name
    
    pipe = Stage1_PriorPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe.prior = Stage1_PriorTransformer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="prior", 
        num_embeddings=2,
        embedding_dim=1024, 
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True
    )

    checkpoint_state_dict = torch.load(model_ckpt, map_location="cpu")

    # 根据checkpoint格式选择正确的处理方式
    if "module" in checkpoint_state_dict:
        # 原始格式的处理方式
        prior_dict = checkpoint_state_dict["module"]
        pipe.prior.load_state_dict(prior_dict)
    else:
        # 新格式的处理方式
        weight_dict = checkpoint_state_dict.get("module", checkpoint_state_dict)
        
        if not any(key.startswith("module.") for key in weight_dict.keys()):
            new_weight_dict = {f"module.{key}": value for key, value in weight_dict.items()}
            pipe.prior.load_state_dict(new_weight_dict)
        else:
            pipe.prior.load_state_dict(weight_dict)
            
    print(f"成功从{model_ckpt}加载模型权重")
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).eval().to(device)

    print('====================== 模型加载完成 ===================')

    # 开始测试
    start_time = time.time()
    number = 0
    sum_simm = 0
    
    # 使用tqdm显示进度
    for batch in tqdm(test_dataloader, desc="处理批次"):
        batch_size = batch["clip_agnostic_img"].size(0)
        img_names = batch.get("img_name", ["unknown.jpg"] * batch_size)
        
        # 将所有输入数据移动到设备上
        clip_agnostic_imgs = batch["clip_agnostic_img"].to(device)
        clip_cloth_imgs = batch["clip_cloth_img"].to(device)
        clip_warp_mask_imgs = batch["clip_warp_mask_img"].to(device)
        clip_image_imgs = batch["clip_image_img"].to(device)
        
        with torch.no_grad():
            # 批量编码图像
            s_agnostic_img_embeds = image_encoder(clip_agnostic_imgs).image_embeds.unsqueeze(1)
            clip_s_image_img_embeds = image_encoder(clip_image_imgs).image_embeds.unsqueeze(1)
            clip_s_cloth_img_embeds = image_encoder(clip_cloth_imgs).image_embeds.unsqueeze(1)
            target_embeds = image_encoder(clip_warp_mask_imgs).image_embeds
            
            # 批量推理
            outputs = pipe(
                s_embed=clip_s_cloth_img_embeds,
                s_pose=s_agnostic_img_embeds,
                t_pose=clip_s_image_img_embeds,
                num_images_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guidance_scale=args.guidance_scale,
            )
            
            predict_embeds = outputs[0]
            
            # 处理每个样本的结果
            for i in range(batch_size):
                number += 1
                predict_embed = predict_embeds[i:i+1]
                target_embed = target_embeds[i:i+1]
                img_name = img_names[i] if isinstance(img_names, list) else img_names
                
                # 保存特征
                feature = predict_embed.cpu().detach().numpy()
                feature_filename = img_name.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")
                np.save(os.path.join(save_dir, feature_filename), feature)
                
                # 计算相似度
                cosine_similarity = F.cosine_similarity(predict_embed, target_embed)
                sum_simm += cosine_similarity.item()

    end_time = time.time()
    print(f"总处理时间: {end_time-start_time}秒")

    avg_simm = sum_simm/number
    with open(save_dir+'/a_results.txt', 'a') as ff:
        ff.write('测试样本数: {}, 引导尺度: {}, 平均相似度: {} \n'.format(number, args.guidance_scale, avg_simm))
    print('测试样本数: {}, 引导尺度: {}, 平均相似度: {}'.format(number, args.guidance_scale, avg_simm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用DataLoader的Stage1先验模型批处理测试脚本")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="kandinsky-community/kandinsky-2-2-prior", 
                      help="预训练模型路径或huggingface.co/models上的模型标识符")
    parser.add_argument("--image_encoder_path", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
                      help="图像编码器路径或huggingface.co/models上的模型标识符")
    parser.add_argument("--img_path", type=str, default="/home/zqc/project/datat/VITON/test", help="图像路径")
    parser.add_argument("--save_path", type=str, default="/home/zqc/project/stage1_test", help="保存路径")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="引导尺度")
    parser.add_argument("--seed_number", type=int, default=42, help="随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument("--img_width", type=int, default=384, help="图像宽度")
    parser.add_argument("--img_height", type=int, default=512, help="图像高度")
    parser.add_argument("--weights_name", type=str, default="/home/zqc/project/weight/stage1-35000.pt", help="权重文件")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader的工作线程数")

    args = parser.parse_args()
    print(args)

    # 设置GPU数量
    num_devices = torch.cuda.device_count()
    print("使用{}个GPU进行推理".format(num_devices))

    # 加载数据
    image_folder = os.path.join(args.img_path, "image")
    image_files = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    select_test_datas = image_files
    print('测试数据数量: {}'.format(len(select_test_datas)))

    # 单卡版本，直接调用main函数
    main(args, 0, select_test_datas)  # rank设为0，使用所有测试数据
    
    # 多卡版本（如需使用，请取消下面的注释）
    # mp.set_start_method("spawn")
    # data_list = split_list_into_chunks(select_test_datas, num_devices)
    # processes = []
    # for rank in range(num_devices):
    #     p = mp.Process(target=main, args=(args, rank, data_list[rank],))
    #     processes.append(p)
    #     p.start()
    # for rank, p in enumerate(processes):
    #     p.join()