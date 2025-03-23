import os
import random
import argparse
import json
import itertools
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
# 读取图片
img_path = "/home/zqc/project/datat/VITON/train/agnostic-v3.2/00000_00.jpg"
class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size


        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        self.order = order

        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()
        self.annotation_pair = []
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # subject_txt = self.txt_preprocess['train']("shirt")
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))

        image = self.transform(im_pil_big)
        # load parsing image


        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-v3.2", im_name)).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        densepose_name = im_name
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", densepose_name)
        )
        pose_img = self.toTensor(densepose_map)  # [-1,1]
 


        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)



            if random.random()>0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

              
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                image = transforms.functional.affine(
                    image, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = transforms.functional.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                pose_img = transforms.functional.affine(
                    pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )



            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                image = transforms.functional.affine(
                    image,
                    angle=0,
                    translate=[shift_valx * image.shape[-1], shift_valy * image.shape[-2]],
                    scale=1,
                    shear=0,
                )
                mask = transforms.functional.affine(
                    mask,
                    angle=0,
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                    scale=1,
                    shear=0,
                )
                pose_img = transforms.functional.affine(
                    pose_img,
                    angle=0,
                    translate=[
                        shift_valx * pose_img.shape[-1],
                        shift_valy * pose_img.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )




        mask = 1-mask

        cloth_trim =  self.clip_processor(images=cloth, return_tensors="pt").pixel_values


        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask

        pose_img =  self.norm(pose_img)


        result = {}
        result["c_name"] = c_name
        result["image"] = image
        result["cloth"] = cloth_trim
        result["cloth_pure"] = self.transform(cloth)
        result["inpaint_mask"] = 1-mask
        result["im_mask"] = im_mask
        result["caption"] = "model is wearing " + cloth_annotation
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["annotation"] = cloth_annotation
        result["pose_img"] = pose_img


        return result

    def __len__(self):
        return len(self.im_names)



def visualize_dataset():
    # 初始化数据集
    dataset = VitonHDDataset(
        dataroot_path="/home/zqc/project/datat/VITON",  # 请确保这是正确的数据根目录
        phase="train",
        order="paired",
        size=(512, 384)
    )
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 获取一个样本
    sample = next(iter(dataloader))
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. 显示原始图像
    original_image = sample['image'][0].permute(1, 2, 0).numpy()
    original_image = (original_image + 1) / 2.0  # 归一化到[0,1]范围
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 2. 显示mask图像
    mask_image = sample['inpaint_mask'][0].permute(1, 2, 0).numpy()
    axes[0, 1].imshow(mask_image, cmap='gray')
    axes[0, 1].set_title('Mask图像')
    axes[0, 1].axis('off')
    
    # 3. 显示服装图像
    cloth_image = sample['cloth_pure'][0].permute(1, 2, 0).numpy()
    cloth_image = (cloth_image + 1) / 2.0  # 归一化到[0,1]范围
    axes[1, 0].imshow(cloth_image)
    axes[1, 0].set_title('服装图像')
    axes[1, 0].axis('off')
    
    # 4. 显示姿态图像
    pose_image = sample['pose_img'][0].permute(1, 2, 0).numpy()
    pose_image = (pose_image + 1) / 2.0  # 归一化到[0,1]范围
    axes[1, 1].imshow(pose_image)
    axes[1, 1].set_title('姿态图像')
    axes[1, 1].axis('off')
    
    # 显示标题
    plt.suptitle(f"样本标注: {sample['annotation'][0]}", y=0.95)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        visualize_dataset()
    except Exception as e:
        print(f"可视化过程中发生错误: {str(e)}")