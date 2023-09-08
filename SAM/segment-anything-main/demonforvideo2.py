import os
import cv2
import numpy as np
import random
import torch

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# 输入必要的参数
model_path = r'L:\DL\SAM\segment-anything-main\models\sam_vit_b_01ec64.pth'
video_path = r'L:\DL\SAM\segment-anything-main\notebooks\videos\input_video.mp4'
output_folder = r"L:\DL\SAM\segment-anything-main\output11"
output_video_path = r"L:\DL\SAM\segment-anything-main\output11\output_video.mp4"

# 设置目标检测模型和权重路径
yolov5_model = "L:\DL\SAM\segment-anything-main\yolov5-master\models\yolov5s.yaml"
yolov5_weights = "L:\DL\SAM\segment-anything-main\yolov5-master\weights\yolov5s.pt"

# 设置目标检测阈值和类别过滤器
confidence_threshold = 0.5
class_filter = ["car", "truck", "bus"]  # 只保留车辆类别

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

# 生成固定颜色列表，用于标记每个类别的目标
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 仅供示例使用，您可以自定义颜色列表

# 创建分割对象
mask_generator = SamAutomaticMaskGenerator(sam)

# 逐帧处理视频
while True:
    # 读取视频帧
    ret, frame = video_capture.read()

    if not ret:
        break

    # 进行目标检测，获取检测结果
    objects = detect(frame, yolov5_model, yolov5_weights, confidence_threshold, class_filter)

    # 进行分割，获取分割结果
    masks = mask_generator.generate(frame)

    # 在原始帧上显示目标检测结果和分割结果
    for obj in objects:
        class_id, class_name, confidence, bbox = obj

        # 绘制边界框和类别标签
        x, y, w, h = bbox
        color = color_list[class_id % len(color_list)]  # 获取固定的颜色
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for mask in masks:
        mask_array = mask['segmentation']
        mask_rgb = np.zeros_like(frame)
        mask_rgb[:, :, 0] = mask_array * 255  # 设置红色通道
        mask_rgb[:, :, 1] = mask_array * 255  # 设置绿色通道
        mask_rgb[:, :, 2] = mask_array * 255  # 设置蓝色通道

        # 将分割结果应用到原始帧上
        frame[mask_array > 0] = mask_rgb[mask_array > 0]

    # 写入输出视频帧
    output_video.write(frame.astype(np.uint8))

# 释放资源
video_capture.release()
output_video.release()

# 删除VideoCapture对象以避免Windows上出现无法删除视频文件的问题
del video_capture
