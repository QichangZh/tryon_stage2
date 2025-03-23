import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取图片
image_path = '/home/zqc/project/datat/VITON/train/agnostic-v3.2/00000_00.jpg'
img = cv2.imread(image_path)

# 将图像从BGR转为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 获取图像的行、列、通道数
row, col, ch = img_rgb.shape

# 设置高斯噪声的均值和标准差
mean = 0
stddev = 100

# 生成与图像同样大小的高斯噪声
gaussian_noise = np.random.normal(mean, stddev, (row, col, ch))

# 将噪声添加到原图像上
noisy_img = np.clip(img_rgb + gaussian_noise, 0, 255).astype(np.uint8)

# 保存加噪声后的图像
output_path = 'noisy_image.jpg'
cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))

# 显示原图和添加噪声后的图像
plt.figure(figsize=(10, 5))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 添加噪声后的图像
plt.subplot(1, 2, 2)
plt.imshow(noisy_img)
plt.title('Image with Gaussian Noise')
plt.axis('off')

plt.show()
