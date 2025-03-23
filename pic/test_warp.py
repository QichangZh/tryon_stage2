import cv2
import numpy as np
import os

# 定义图片文件夹路径
image_folder = '/home/zqc/project/datat/VITON/train/agnostic-v3.2/'
mask_folder = '/home/zqc/project/datat/VITON/train/mask1/'

# 如果目标文件夹不存在，创建它
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历文件夹中的每一张图片
for image_file in image_files:
    # 读取图片
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # 转换为 HSV 并创建二值掩码
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 128])
    upper_gray = np.array([0, 0, 128])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # 生成保存掩码的文件路径
    mask_path = os.path.join(mask_folder, f"{image_file}")
    # inverted_image = cv2.bitwise_not(mask)

    # 保存掩码
    cv2.imwrite(mask_path, mask)