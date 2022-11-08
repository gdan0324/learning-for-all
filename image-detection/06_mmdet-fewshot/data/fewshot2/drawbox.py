import os
import cv2
import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np

# 修改
IMG_PATH = "C:\\Users\\96212\\Desktop\\abnorm\\test\\images"
TXT_PATH = "C:\\Users\\96212\\Desktop\\detection-results1"
OUT_PATH = "C:\\Users\\96212\\Desktop\\vi1"
########

if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

for images in os.listdir(IMG_PATH):

    with open(os.path.join(TXT_PATH, os.path.splitext(images)[0] + '.txt'), 'r') as file:
        lines = file.readlines()
        # 1.读取图片
        im = Image.open(os.path.join(IMG_PATH, images))
        draw = ImageDraw.Draw(im)
        for line in lines:
            # 2.获取标签 box_list
            content_list = line.split()
            draw.rectangle([int(content_list[2]), int(content_list[3]), int(content_list[4]), int(content_list[5])],
                           outline='red', width=1)  # 画bbox
            font = ImageFont.truetype("consola.ttf", 12, encoding="unic")  # 设置字体
            draw.text((int(content_list[2]) - 15, int(content_list[3]) - 15), content_list[0], 'green', font)  # 写label
            draw.text((int(content_list[2]) + 30, int(content_list[3]) - 15), content_list[1], 'red', font)  # 写label
        del draw
        # 3.保存图片
        im.save(os.path.join(OUT_PATH, images))  # 保存文件
