# coding:utf-8
import numpy as np
import json
import pandas as pd
import os


def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


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
    while order.size > 0:
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


if __name__ == "__main__":

    # 更改
    JSON_PATH = "C:\\Users\\96212\\Desktop\\json"
    OUT_PATH = "C:\\Users\\96212\\Desktop\\detection-results"
    thresh = 0.4
    alltest_path = 'C:\\Users\\96212\\Desktop\\汇报\\科大讯飞赛题\\小样本\\data\\testjpg'
    #####

    # 创建保存路径
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    # 当前目录下的json文件
    jsonlist = [os.path.join(JSON_PATH, img) for img in os.listdir(JSON_PATH)]

    # 加载第一个文件
    with open(jsonlist[0]) as f:
        load_dic = json.load(f)

    # 加载后面的对比文件
    for jsonpath in jsonlist[1:]:
        with open(jsonpath) as f:
            temp_dic = json.load(f)
            for k in load_dic.keys():
                load_dic[k] += temp_dic[k]

    # 保存最终的投票结果
    for picname in load_dic.keys():
        boxes = load_dic[picname]
        if len(boxes) > 1:
            n = np.array(boxes[0])
            for box in boxes[1:]:
                n = np.vstack((n, np.array(box)))
            keep = py_cpu_nms(n, thresh)
            keep = keep.tolist()
            keep.sort(key=lambda x: x[0])

            for b in keep:
                with open(os.path.join(OUT_PATH, os.path.splitext(picname)[0] + '.txt'), 'a') as f:
                    f.write("defect" + " " + str(b[4]) + " " + str(int(b[0])) + " " + str(int(b[1])) + " " + str(
                        int(b[2])) + " " + str(int(b[3])) + '\n')

        else:
            for b in boxes:
                with open(os.path.join(OUT_PATH, os.path.splitext(picname)[0] + '.txt'), 'a') as f:
                    f.write("defect" + " " + str(b[4]) + " " + str(int(b[0])) + " " + str(int(b[1])) + " " + str(
                        int(b[2])) + " " + str(int(b[3])) + '\n')

    # 将正常文件给空文件
    file_test_lists = [os.path.splitext(img)[0] for img in os.listdir(alltest_path)]
    out_lists = [os.path.splitext(txt)[0] for txt in os.listdir(OUT_PATH)]
    for fileitem in file_test_lists:
        if fileitem not in out_lists:
            # 正常图片直接给空
            emptyfile = fileitem + '.txt'
            with open(os.path.join(OUT_PATH, emptyfile), 'w') as file:
                pass

    print("finish")
