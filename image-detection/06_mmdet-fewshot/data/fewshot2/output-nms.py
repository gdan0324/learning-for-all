from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import pandas as pd
import json
import torch
import numpy as np

config_file = r'E:\DL\DL-objectdetect\mmdet-fewshot\fewshot2\cbnetv2_template\cbnetv2_temp.py'
checkpoint_file = r'E:\DL\DL-objectdetect\mmdet-fewshot\fewshot2\cbnetv2_template\epoch_72.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

d = {}
image_path = "C:\\Users\\96212\\Desktop\\abnorm\\test\\images"
save_path = 'C:\\Users\\96212\\Desktop\\detection-results'
piclist = os.listdir(image_path)

if not os.path.isdir(save_path):
    os.mkdir(save_path)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0.0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return dets[keep]


for pic_name in piclist:
    pic_path = os.path.join(image_path, pic_name)
    # print(pic_name + ':')
    result = inference_detector(model, pic_path)
    # print(result[0])
    # result = [py_cpu_nms(result[0], 0.1)]
    # print(keep)
    boxes = []
    # print(result)
    for i in range(1):
        for box in result[i]:
            # 转换成列表
            cbox = []
            copybox = box.tolist()

            if i == 0:
                copybox.append('defect')

            # print(copybox)
            cbox.append('defect')
            cbox.append(copybox[4])
            cbox.extend(copybox[:4])

            # 置信度
            if copybox[-2] >= 0.1:
                boxes.append(cbox)
            print(copybox[-2])
    boxes.sort(key=lambda x: x[0])
    # print(boxes)

    f_name = pic_name.split(".")[0] + ".txt"
    # print(os.path.join(save_path, f_name))
    f = open(os.path.join(save_path, f_name), 'w')
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            if j == 0:
                f.write(str(boxes[i][j]) + " ")
            elif j == 1:
                f.write(str(round(boxes[i][j], 6)) + " ")
            elif j != 5:
                f.write(str(int(boxes[i][j])) + " ")
            else:
                f.write(str(int(boxes[i][j])))
        f.write('\n')
    f.close()
