import cv2
import numpy as np
import os

# 定义图片和掩码文件夹路径
image_folder = '/home/zqc/project/datat/VITON/train/image/'
mask_folder = '/home/zqc/project/datat/VITON/train/mask1/'
output_folder = '/home/zqc/project/datat/VITON/train/black_cloth/'

# 如果目标文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有图片文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历文件夹中的每一张图片
for image_file in image_files:
    # 构建图像和掩码的完整路径
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, image_file)

    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 确保mask为二值图像（黑白图像），即只有0和255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 创建一个全黑的图像，与原始图像大小相同
    black_image = np.zeros_like(image)

    # 将图像中的衣服部位（mask中为白色的部分）替换为黑色
    image_with_black_clothes = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    # 将图像中的其他部分（mask中为黑色的部分）保持不变
    image_with_black_clothes += cv2.bitwise_and(black_image, black_image, mask=mask)

    # 保存结果
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image_with_black_clothes)
