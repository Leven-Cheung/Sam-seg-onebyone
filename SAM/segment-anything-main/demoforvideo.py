import os
import cv2
import numpy as np
import random
import torch

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# 随机生成颜色
def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (b, g, r)

# 输入必要的参数
model_path = r'L:\DL\SAM\segment-anything-main\models\sam_vit_b_01ec64.pth'
video_path = r'L:\DL\SAM\segment-anything-main\notebooks\images\video1.mp4'
output_folder = r"L:\DL\SAM\segment-anything-main\output1video"
output_video_path = r"L:\DL\SAM\segment-anything-main\outputvideo\output_video.mp4"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 加载模型
sam = sam_model_registry["vit_b"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam.to(device)

# 打开视频文件
video_capture = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建用于保存输出视频的编码器和写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# 逐帧处理视频
while True:
    # 读取视频帧
    ret, frame = video_capture.read()

    if not ret:
        break

    # 进行预测，生成掩码
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(frame)

    # 在原始帧上显示分割的物体
    for mask in masks:
        mask_array = mask['segmentation']
        color = generate_random_color()
        frame[mask_array > 0] = color

    # 写入输出视频帧
    output_video.write(frame.astype(np.uint8))

# 释放资源
video_capture.release()
output_video.release()