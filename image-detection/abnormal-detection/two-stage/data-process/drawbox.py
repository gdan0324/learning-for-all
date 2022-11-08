import os
import cv2
import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np
import xml.etree.ElementTree as ET

# 修改
IMG_PATH = "../stage5/data/train/images/"
XML_PATH = "../stage5/data/train/Annotations/"
OUT_PATH = "../stage5/data/train/box/"
########

if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

xml_files = os.listdir(XML_PATH)
for file in xml_files:
        xml_file_path = os.path.join(XML_PATH, file)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find("size")
        # 获取xml的width和height的值
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        # object标签可能会存在多个，所以要迭代
        # 1.读取图片
        idx = os.path.splitext(file)[0]
        im = Image.open(os.path.join(IMG_PATH,idx + '.png'))
        draw = ImageDraw.Draw(im)
        for obj in root.iter("object"):
                difficult = obj.find("difficult").text
                # 种类类别
                cls = obj.find("name").text
                xml_box = obj.find("bndbox")
                box = (float(xml_box.find("xmin").text), float(xml_box.find("ymin").text),
                        float(xml_box.find("xmax").text), float(xml_box.find("ymax").text))
                draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])],outline='red',width=1)  # 画bbox
                font = ImageFont.truetype("consola.ttf", 12, encoding="unic" )  # 设置字体
                draw.text((int(box[0])-15, int(box[1])-15),str(cls), 'green', font)    # 写label
        del draw
        # 3.保存图片
        im.save(os.path.join(OUT_PATH, idx + '.png'))   # 保存文件