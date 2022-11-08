import os
import shutil
import warnings
import json
from PIL import Image, ImageDraw
from PIL import ImageFont

warnings.filterwarnings('ignore')

# 改一下 BASE_PATH 运行三次
BASE_PATH = "C:\\Users\\96212\\Desktop\\官网数据+打上异常图框+json训练集图片\\train\\03"

IMG_TRAIN_PATH = os.path.join(BASE_PATH, "Images")
XML_PATH = os.path.join(BASE_PATH, "Annotations")
ABNORM_IMG_PATH = os.path.join(BASE_PATH, "abnormImages")
ABNORM_LABEL_PATH = os.path.join(BASE_PATH, "abnormImagesLabel")

IMAGE_TYPE = '.png'


def pil_draw(img_path, abnormal_file, outlier_file, pic_type):
    print(img_path)
    # 读取json文件
    with open(os.path.join(img_path, 'TRAIN_objects.json'), 'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict)
    for item in load_dict:
        # print(item)
        # 1.读取图片
        im = Image.open(os.path.join(abnormal_file, item['name'] + pic_type))
        # 2.获取标签 box_list
        draw = ImageDraw.Draw(im)
        for box_list, label in zip(item['boxes'], item['labels']):
            # print(box_list)
            draw.rectangle([box_list[0], box_list[1], box_list[2], box_list[3]], outline='red', width=1)  # 画bbox
            font = ImageFont.truetype("consola.ttf", 12, encoding="unic")  # 设置字体
            draw.text((box_list[0] - 15, box_list[1] - 15), label, 'green', font)  # 写label
        del draw
        # 3.保存图片
        isExists = os.path.exists(outlier_file)
        if not isExists:
            os.makedirs(outlier_file)  # 创建文件路径
        else:
            im.save(os.path.join(outlier_file, item['name'] + pic_type))  # 保存文件
    print('保存成功')


outlier_file = ABNORM_LABEL_PATH

# 第二类数据有点奇怪，给的都是绿色，出来不是
pil_draw(BASE_PATH, ABNORM_IMG_PATH, outlier_file, IMAGE_TYPE)
