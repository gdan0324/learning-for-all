"""python
    xml文件解析
"""

import json
import os
import torch
import random
import xml.etree.ElementTree as ET  # 解析xml文件所用工具
import torchvision.transforms.functional as FT
import sys


def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


# 解析xml文件，最终返回这张图片中所有目标的标注框及其类别信息，以及这个目标是否是一个difficult目标
def parse_annotation(annotation_path, name):
    # 解析xml
    tree = ET.parse(os.path.join(annotation_path, name))
    tree_name = os.path.splitext(name)[0]
    root = tree.getroot()

    boxes = list()  # 存储bbox  分别表示目标的左上角和右下角坐标
    labels = list()  # 存储bbox对应的label  目标类别
    difficulties = list()  # 存储bbox对应的difficult信息  表示此目标是否是一个难以识别的目标

    # 遍历xml文件中所有的object，有多少个object就有多少个目标
    for object in root.iter('object'):
        # 提取每个object的difficult、label、bbox信息
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()  ## strip去除首尾的空格
        # label = 'defect'
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        # 存储
        boxes.append([xmin, ymin, xmax, ymax])
        # labels.append(annotation_path[-11:-9])
        labels.append(label)
        difficulties.append(difficult)

    # 返回包含图片标注信息的字典
    return {'name': tree_name, 'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(path, output_folder):
    # 获取数据集的绝对路径
    path = os.path.abspath(path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    file_name = getFileList(path)
    print(file_name)
    # 根据图片id，解析图片的xml文件，获取标注信息
    for id in file_name:
        # Parse annotation's XML file
        objects = parse_annotation(path, id)
        print(os.path.join(path, id))
        print(objects)
        if len(objects['boxes']) == 0:  # 如果没有目标则跳过
            continue
        n_objects += len(objects['boxes'])  # 统计目标总数
        train_objects.append(objects)  # 存储每张图片的标注信息到列表train_objects

    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, fp=j)


if __name__ == "__main__":
    # label文件夹
    xml_path = 'C:\\Users\\96212\\Desktop\\abnorm\\train\\Annotations'
    # 输出结果位置
    output_folder = 'C:\\Users\\96212\\Desktop\\abnorm\\train'

    create_data_lists(path=xml_path, output_folder=output_folder)
