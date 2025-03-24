import os
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob

from transformers import CLIPImageProcessor


def InpaintCollate_fn(data):
 
    target_image = torch.stack([example["clip_t_img"] for example in data])
    target_image = target_image.to(memory_format=torch.contiguous_format).float()

    cloth_image = torch.stack([example["clip_cloth_img"] for example in data])
    cloth_image = cloth_image.to(memory_format=torch.contiguous_format).float()

    warp_image = torch.stack([example["clip_warp_img"] for example in data])
    warp_image = warp_image.to(memory_format=torch.contiguous_format).float()
    

    vae_train_image = torch.stack([example["trans_train_img_mask"] for example in data])
    vae_train_image = vae_train_image.to(memory_format=torch.contiguous_format).float()


    source_target_image = torch.stack([example["trans_st_img"] for example in data])
    source_target_image = source_target_image.to(memory_format=torch.contiguous_format).float()

    vae_mask_img = torch.stack([example["mask_img"] for example in data])
    vae_mask_img = vae_mask_img.to(memory_format=torch.contiguous_format).float()



    return {
        "target_image": target_image, 
        "cloth_image": cloth_image,
        "warp_image" : warp_image,
        "vae_source_mask_image": vae_train_image,
        "source_target_image": source_target_image,
        "vae_mask_img": vae_mask_img,
    }



class InpaintDataset(Dataset):
    def __init__(
        self,
        # json_file,
        image_root_path,
        size=(512,512),   
        imgp_drop_rate=0.0,
        imgg_drop_rate=0.0,
    ):
        self.image_root_path = image_root_path


        self.size = size

        self.imgp_drop_rate = imgp_drop_rate
        self.imgg_drop_rate = imgg_drop_rate

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
        )

        self.image_folder = os.path.join(image_root_path, "image")
        self.image_files = []

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))

        self.image_files = [os.path.basename(f) for f in self.image_files]
        # 打印找到的文件数
        print(f"找到 {len(self.image_files)} 个图像文件")


    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        cloth_img_path = os.path.join(self.image_root_path, "cloth", img_name)
        cloth_img = Image.open(cloth_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        warp_img_path = os.path.join(self.image_root_path, "warp_mask", img_name)
        warp_img = Image.open(warp_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        balck_img_path = os.path.join(self.image_root_path, "black_cloth", img_name)
        black_img = Image.open(balck_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        train_img_mask = Image.new("RGB", (self.size[0] * 2, self.size[1]))
        train_img_mask.paste(cloth_img, (0, 0))
        train_img_mask.paste(black_img, (self.size[0], 0))

        t_img_path = os.path.join(self.image_root_path, "image", img_name)
        t_img = Image.open(t_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        st_img = (Image.new("RGB", (self.size[0] * 2, self.size[1])))
        st_img.paste(cloth_img, (0, 0))
        st_img.paste(t_img, (self.size[0], 0))

        mask_img_path = os.path.join(self.image_root_path, "mask", img_name)
        mask_img = Image.open(mask_img_path).convert("RGB").resize(self.size, Image.BICUBIC)
        # mask_img = mask_img[:1]
        mask_img = self.transform(mask_img)

        # 创建姿态图的并排组合
        # s_pose = Image.open(s_img_path.replace("/train_all_png/", "/openpose_all_img/").replace(".png", "_pose.jpg")).convert("RGB").resize(self.size, Image.BICUBIC)
        # t_pose = Image.open(t_img_path.replace("/train_all_png/", "/openpose_all_img/").replace(".png", "_pose.jpg")).convert("RGB").resize(self.size, Image.BICUBIC)
        # st_pose = Image.new("RGB", (self.size[0] * 2, self.size[1]))
        # st_pose.paste(s_pose, (0, 0))
        # st_pose.paste(t_pose, (self.size[0], 0))


        trans_train_img_mask = self.transform(train_img_mask)
        trans_st_img = self.transform(st_img)
        # trans_st_pose = self.transform(st_pose)


        clip_t_img = (self.clip_image_processor(images=t_img, return_tensors="pt").pixel_values).squeeze(dim=0)
        clip_cloth_img = (self.clip_image_processor(images=cloth_img, return_tensors="pt").pixel_values).squeeze(dim=0)
        clip_warp_img = (self.clip_image_processor(images=warp_img, return_tensors="pt").pixel_values).squeeze(dim=0)

        ## dropout s_img for dinov2
        if random.random() < self.imgp_drop_rate:
            clip_cloth_img = torch.zeros(clip_cloth_img.shape)

        ## dropout t_img_embed
        if random.random() < self.imgg_drop_rate:
            clip_warp_img = torch.zeros(clip_warp_img.shape)

        return {
            # "clip_s_img": clip_s_img,
            "clip_t_img": clip_t_img,
            "clip_cloth_img": clip_cloth_img, 
            "clip_warp_img": clip_warp_img,
            "trans_st_img": trans_st_img,
            "trans_train_img_mask": trans_train_img_mask,
            "mask_img": mask_img,
        }

    def __len__(self):
        return len(self.image_files)



def test_black_img_channels():
    """
    测试函数：提取black_img的第一个通道并可视化
    """
    import numpy as np
    
    # 指定测试数据路径
    test_path = "/home/zqc/project/datat/VITON/train/"
    
    # 初始化数据集
    dataset = InpaintDataset(image_root_path=test_path)
    
    # 获取第一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        
        # 获取原始图像
        mask_img = sample["mask_img"]
        
        # 转换为numpy数组
        mask_img_np = np.array(mask_img)
        print(f"原始black_img形状: {mask_img_np.shape}")
        
        # 提取第一个通道
        first_channel = mask_img_np[:,:,0]
        print(f"第一个通道形状: {first_channel.shape}")
        
        # 可视化
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            
            # 显示原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(mask_img)
            plt.title("原始图像")
            
            # 显示第一个通道
            plt.subplot(1, 2, 2)
            plt.imshow(first_channel, cmap='gray')
            plt.title("第一个通道 (R)")
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("无法导入matplotlib显示图像")
    else:
        print("数据集为空，无法测试")

if __name__ == "__main__":
    test_black_img_channels()