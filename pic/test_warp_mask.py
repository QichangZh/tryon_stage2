import cv2
import numpy as np
import os

# 定义文件夹路径
mask_dir = "/home/zqc/project/datat/VITON/train/agnostic-v3.2"    # 掩码图像所在目录
image_dir = "/home/zqc/project/datat/VITON/train/image"         # 原始图像所在目录
output_dir = "/home/zqc/project/datat/VITON/train/warp_mask"            # 输出目录

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历掩码目录中的所有图像文件
for filename in os.listdir(mask_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        mask_path = os.path.join(mask_dir, filename)
        image_path = os.path.join(image_dir, filename)  # 假设两文件夹中的文件名一致

        # 读取图像
        mask_img = cv2.imread(mask_path)
        main_img = cv2.imread(image_path)

        if mask_img is None or main_img is None:
            print(f"跳过文件 {filename}，无法加载图像。")
            continue

        # 转换掩码图像到 HSV，并生成二值掩码
        hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
        # 此处的阈值用于匹配灰色区域，可根据实际情况调整
        lower_gray = np.array([0, 0, 128])
        upper_gray = np.array([0, 0, 128])
        binary_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("wrong")
            continue
        
        # 创建掩码并绘制轮廓
        contour_mask = np.zeros_like(main_img, dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # 保留掩码区域，其他区域为黑色
        mask_only_region = cv2.bitwise_and(main_img, contour_mask)

        # 创建最终图像：将非掩码区域设为黑色，掩码区域保留
        final_image = np.copy(main_img)
        final_image[contour_mask == 0] = 0  # 将非掩码区域设为黑色
        
        # 保存最终图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, final_image)
        print(f"处理完成并保存: {output_path}")
