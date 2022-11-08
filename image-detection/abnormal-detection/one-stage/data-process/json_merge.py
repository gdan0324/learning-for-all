"""
    得到模型融合的json文件
"""

import json
import os


def parse_txt(name):
    with open(os.path.join(IMG_PATH, name), 'r') as file:
        lines = file.readlines()
        content = list()  # 存储bbox  分别表示目标的左上角和右下角坐标
        for line in lines:
            content_list = line.split()
            content.append([int(content_list[2]), int(content_list[3]), int(content_list[4]), int(content_list[5]),
                            float(content_list[1]), 0])
    return content


if __name__ == "__main__":

    # 更改
    # 检测得到的label路径，必须是完整的三类！
    IMG_PATH = "C:\\Users\\96212\\Desktop\\detection-results"

    # json保存路径
    JSON_PATH = "C:\\Users\\96212\\Desktop\\json"
    #####

    # 创建保存路径
    if not os.path.isdir(JSON_PATH):
        os.mkdir(JSON_PATH)

    # 得到保存文件的内容
    res = {}
    for images in os.listdir(IMG_PATH):
        objects = parse_txt(images)
        res[images] = objects

    # 保存
    with open(os.path.join(JSON_PATH, 'json_list.json'), 'a') as j:
        json.dump(res, fp=j)

    if res:
        print('finish')
    else:
        print('error')


