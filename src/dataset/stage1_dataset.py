import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor
import glob

def PriorCollate_fn(batch):
    """
    自定义的collate函数，用于处理PriorImageDataset返回的字典格式数据
    将batch中的样本合并成一个批次
    """
    clip_agnostic_imgs = torch.stack([item["clip_agnostic_img"] for item in batch])
    clip_cloth_imgs = torch.stack([item["clip_cloth_img"] for item in batch])
    clip_warp_mask_imgs = torch.stack([item["clip_warp_mask_img"] for item in batch])
    clip_image_imgs = torch.stack([item["clip_image_img"] for item in batch])
    
    return {
            "clip_agnostic_img": clip_agnostic_imgs,
            "clip_cloth_img": clip_cloth_imgs,
            "clip_warp_mask_img": clip_warp_mask_imgs,
            "clip_image_img": clip_image_imgs,
        }


class PriorImageDataset(Dataset):
    def __init__(
        self,
        # json_file,
        size=(512, 512),
        s_img_drop_rate=0.0,
        t_img_drop_rate=0.0,
        s_pose_drop_rate=0.0,
        t_pose_drop_rate=0.0,
        image_root_path=""
    ):
        """
        Args:
            size: 需要 resize 到的图像大小
            s_img_drop_rate, t_img_drop_rate: 随机丢弃 source/target 图像的概率
            s_pose_drop_rate, t_pose_drop_rate: 随机丢弃 source/target pose 的概率
            image_root_path: 数据根目录，比如 "/data"
        """
        # self.data = json.load(open(json_file, "r"))
        self.size = size
        self.s_img_drop_rate = s_img_drop_rate
        self.t_img_drop_rate = t_img_drop_rate
        self.s_pose_drop_rate = s_pose_drop_rate
        self.t_pose_drop_rate = t_pose_drop_rate
        self.image_root_path = image_root_path

        self.image_folder = os.path.join(image_root_path, "image")
        self.image_files = []

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))

        self.image_files = [os.path.basename(f) for f in self.image_files]
        # 打印找到的文件数
        print(f"找到 {len(self.image_files)} 个图像文件")

        # CLIP 的图像预处理器
        self.clip_image_processor = CLIPImageProcessor()

        # 可选的普通 transforms（如果需要）
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # item = self.data[idx]

        # img_name = item["source_image"]
        # t_img_name = item["target_image"].replace(".jpg", ".png")

        img_name = self.image_files[idx]

        # ---------------------------
        # 2. 从四个子文件夹中读取 source 的图片
        # ---------------------------
        s_agnostic_path = os.path.join(self.image_root_path, "agnostic-v3.2", img_name)
        s_cloth_path    = os.path.join(self.image_root_path, "cloth",        img_name)
        s_warp_mask_path = os.path.join(self.image_root_path, "warp_mask", img_name)
        s_image_path    = os.path.join(self.image_root_path, "image",        img_name)

        # 依次打开并 resize
        s_agnostic_img = Image.open(s_agnostic_path).convert("RGB").resize(self.size, Image.BICUBIC)
        s_cloth_img    = Image.open(s_cloth_path).convert("RGB").resize(self.size, Image.BICUBIC)
        s_warp_mask_img = Image.open(s_warp_mask_path).convert("RGB").resize(self.size, Image.BICUBIC)
        s_image_img    = Image.open(s_image_path).convert("RGB").resize(self.size, Image.BICUBIC)



        # ---------------------------
        # 3. 使用 CLIPImageProcessor 对主要图像进行处理
        #    这里示例仅对 s_image_img 和 t_image_img 做 CLIP 的处理，
        #    如果需要对其它图像（如 cloth、agnostic 等）做同样处理，可类比扩展。
        # ---------------------------
        clip_agnostic_img = self.clip_image_processor(images=s_agnostic_img, return_tensors="pt").pixel_values.squeeze(0)
        clip_cloth_img = self.clip_image_processor(images=s_cloth_img, return_tensors="pt").pixel_values.squeeze(0)
        clip_warp_mask_img = self.clip_image_processor(images=s_warp_mask_img, return_tensors="pt").pixel_values.squeeze(0)
        clip_image_img = self.clip_image_processor(images=s_image_img, return_tensors="pt").pixel_values.squeeze(0)


        # ---------------------------
        # 6. 随机丢弃
        #    根据你的需求，这里保留最初的逻辑
        # ---------------------------
        if random.random() < self.s_img_drop_rate:
            clip_agnostic_img = torch.zeros_like(clip_agnostic_img)
        if random.random() < self.s_pose_drop_rate:
            clip_cloth_img = torch.zeros_like(clip_cloth_img)
        if random.random() < self.t_img_drop_rate:
            clip_warp_mask_img = torch.zeros_like(clip_warp_mask_img)
        if random.random() < self.t_pose_drop_rate:
            clip_image_img = torch.zeros_like(clip_image_img)

        return {
            "clip_agnostic_img": clip_agnostic_img,
            "clip_cloth_img": clip_cloth_img,
            "clip_warp_mask_img": clip_warp_mask_img,
            "clip_image_img": clip_image_img,
        }

        # 
