import os
import cv2

file_dir = 'L:/DL/SAM/esult/output10/'
file_list = []

for root, dirs, files in os.walk(file_dir):
    for file in files:
        file_list.append(file)

video = cv2.VideoWriter('L:/DL/SAM/segment-anything-main/outputvideo/test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 5, (1280, 720))

# 添加照片1
image_path1 = 'L:/DL/SAM/segment-anything-main/notebooks/images/test01.jpg'
image1 = cv2.imread(image_path1)
if image1 is not None:
    image1 = cv2.resize(image1, (1280, 720))
    for _ in range(3 * 5):  # 显示 3 秒钟，每秒 5 帧
        video.write(image1)

# 添加照片2
image_path2 = 'L:/DL/SAM/esult/output10/output_image.png'
image2 = cv2.imread(image_path2)
if image2 is not None:
    image2 = cv2.resize(image2, (1280, 720))
    for _ in range(3 * 5):  # 显示 3 秒钟，每秒 5 帧
        video.write(image2)

# 添加照片3
image_path3 = 'L:/DL/SAM/esult/output11/output_combined_image.png'
image3 = cv2.imread(image_path3)
if image3 is not None:
    image3 = cv2.resize(image3, (1280, 720))
    for _ in range(3 * 5):  # 显示 3 秒钟，每秒 5 帧
        video.write(image3)

for i in range(1, len(file_list)):
    img = cv2.imread(file_dir + file_list[i-1])
    if img is not None:
        img = cv2.resize(img, (1280, 720))
        video.write(img)

video.release()
