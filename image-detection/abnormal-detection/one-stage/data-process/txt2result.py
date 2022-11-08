import os
import json
from PIL import Image, ImageDraw
import pandas as pd
import torch
import numpy as np
import sys


# 读取txt文件
def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


# 获取 image-size
def getImageSize(path, filename, type):
    image = Image.open(os.path.join(path, filename + type))
    return image.size


# box [cx,cy,w,h]
def convert(size, box):
    # 解宽高归一化
    x_center = box[0] * size[0]
    y_center = box[1] * size[1]

    xmin = (2 * x_center - box[2] * size[0]) / 2.0
    xmax = (2 * x_center + box[2] * size[0]) / 2.0

    ymin = (2 * y_center - box[3] * size[1]) / 2.0
    ymax = (2 * y_center + box[3] * size[1]) / 2.0

    #     xmin = int(((float(box[0]))*size[0]+1)-(float(box[2]))*0.5*size[0])
    #     ymin = int(((float(box[1]))*size[1]+1)-(float(box[3]))*0.5*size[1])

    #     xmax = int(((float(box[0]))*size[0]+1)+(float(box[2]))*0.5*size[0])
    #     ymax = int(((float(box[1]))*size[1]+1)+(float(box[3]))*0.5*size[1])

    return (int(xmin), int(ymin), int(xmax), int(ymax))


# 生成结果文件
def pro_result(file_test_lists, file_name, file_path, test_path, result_path):
    """
        file_test_lists: 所有测试图片的名字
        file_name: 从yolo检测到异常的label
        file_path: 异常图片的label的path位置
        file_test: 所有测试图片的图片的path位置
        result_path: 输出文件位置
    """
    isExists = os.path.exists(result_path)
    if not isExists:
        os.makedirs(result_path)  # 创建文件路径
    for fileitem in file_test_lists:
        testfile = os.path.splitext(fileitem)[0]
        type = os.path.splitext(fileitem)[1]
        # print(testfile)
        if testfile + '.txt' in file_name:
            abnormfile = testfile + '.txt'
            # 读取txt文件
            # print(abnormfile)
            with open(os.path.join(file_path, abnormfile), 'r') as load_f:
                for item in load_f:
                    # txt内容
                    item_args = item.split()
                    img_w, img_h = getImageSize(test_path, os.path.splitext(abnormfile)[0], type)
                    box = (float(item_args[1]), float(item_args[2]),
                           float(item_args[3]), float(item_args[4]))

                    # 转换为 box [xmin,ymin,xmax,ymax]
                    boxex = convert((img_w, img_h), box)

                    # 写到一个新的文件里面
                    with open(os.path.join(result_path, abnormfile), 'a') as file:
                        file.write("defect" + " " + str(round(float(item_args[5]), 2)) + " " + " ".join(
                            [str(s) for s in boxex]) + '\n')
                        # file.write(item_args[0] + " " + " ".join([str(s) for s in box]) + '\n')
        else:
            # 正常图片直接给空
            emptyfile = testfile + '.txt'
            with open(os.path.join(result_path, emptyfile), 'w') as file:
                pass


if __name__ == "__main__":
    # 更改
    # 检测出60张中的异常图片的label文件夹
    anlables_path = sys.argv[1]
    # 所有测试文件夹，要求所有都是jpg文件
    alltest_path = sys.argv[2]
    # 输出结果目录
    result_path = sys.argv[3]
    #####

    # 异常图片文件名字
    anfile_names = getFileList(anlables_path)
    # 读取测试所有文件
    file_test_lists = getFileList(alltest_path)

    pro_result(file_test_lists, anfile_names, anlables_path, alltest_path, result_path)
