import os
import cv2
import numpy as np
import random
import torch
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# 输入必要的参数
model_path = r'/home/zgz/DL/segment-anything-main/models/sam_vit_h_4b8939.pth'
image_path = r'/home/zgz/DL/segment-anything-main/notebooks/images/test02.jpg'
output_folder = r"/home/zgz/DL/segment-anything-main/output20"
output_image_path = r"/home/zgz/DL/segment-anything-main/output20/output_image.png"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 加载模型
sam = sam_model_registry["vit_h"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam.to(device)

# 加载图片
image = cv2.imread(image_path)

# 这里是预测，不用提示词，进行全图分割
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 保存掩码
mask_colors = []
for i, mask in enumerate(masks):
    mask_array = mask['segmentation']

    # 生成随机颜色
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    mask_rgb = np.zeros_like(image)
    mask_rgb[:, :, 0] = mask_array * color[0]  # 设置红色通道
    mask_rgb[:, :, 1] = mask_array * color[1]  # 设置绿色通道
    mask_rgb[:, :, 2] = mask_array * color[2]  # 设置蓝色通道

    # 设置颜色的透明度
    mask_alpha = np.where(mask_array > 0, 0.5, 0)  # 设置透明度为50%（0.5）

    # 将掩码应用到原始图像上
    mask_alpha_expanded = np.expand_dims(mask_alpha, axis=2)  # 扩展 mask_alpha 的维度
    masked_image = image * (1 - mask_alpha_expanded) + mask_rgb * mask_alpha_expanded

    # 保存掩码
    output_file = os.path.join(output_folder, f"mask_{i + 1}.png")
    cv2.imwrite(output_file, masked_image)

    # 保存颜色和透明度
    mask_colors.append((mask_rgb, mask_alpha))

# 在原始图像上显示分割的物体
output_image = np.zeros_like(image, dtype=np.float32)
for color, alpha in mask_colors:
    output_image += color * np.expand_dims(alpha, axis=2)

# 将输出图像转换为NumPy数组并进行类型转换
output_image = np.clip(output_image, 0, 255).astype(np.uint8)

# 保存最终输出图像
cv2.imwrite(output_image_path, output_image)
