import os
import numpy as np
import cv2
import torch
import torchvision
import argparse

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings

warnings.filterwarnings('ignore')

# PRED_PATH = "../../../../test/01/"

# RESULT_PATH = 'runs/detect3'
# RESULT_LABEL_PATH = 'runs/detect3/labels'

# make dictionary for class objects so we can call objects by their keys.
classes = {1: 'air-hole', 2: 'hollow-bead', 3: 'slag-inclusion'}
# classes= {1:'defect'}

num_classes = len(classes) + 1  # 多一个背景类

# -------------------------参数-------------------------------
parser = argparse.ArgumentParser(description="detect object")
parser.add_argument('--weights', nargs='+', type=str, default='models/faster_rcnn_state11.pth', help='model.pt path(s)')
parser.add_argument('--source', type=str, default=r'../../../../test/01/')
parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
parser.add_argument('--project', default='runs/detect3', help='save results to project/name')
args = parser.parse_args()

# ------------------------创建结果文件夹----------------------------------
isExists = os.path.exists(args.project)
if not isExists:
    os.makedirs(args.project)  # 创建文件路径

# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load  a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)


# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the traines weights
model.load_state_dict(torch.load(args.weights))

model = model.to(device)


def obj_detector(img):
    img = os.path.join(args.source, img)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    # 转成tensor的形式，浅拷贝
    img = torch.from_numpy(img)
    # 加batch-size
    img = img.unsqueeze(0)
    # batch-size-通道-宽-高
    img = img.permute(0, 3, 1, 2)

    # 进入验证模式
    model.eval()

    # 阈值
    detection_threshold = args.conf_thres

    img = list(im.to(device) for im in img)
    # 送进网络
    output = model(img)

    for i, im in enumerate(img):
        boxes = output[i]['boxes'].data.cpu().numpy()
        scores = output[i]['scores'].data.cpu().numpy()
        labels = output[i]['labels'].data.cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    sample = img[0].permute(1, 2, 0).cpu().numpy()
    sample = np.array(sample)
    boxes = output[0]['boxes'].data.cpu().numpy()
    name = output[0]['labels'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    scores = [i for i in scores if i > detection_threshold]

    names = name.tolist()

    return names, boxes, scores, sample


for images in os.listdir(args.source):
    names, boxes, scores, sample = obj_detector(images)
    i = 0
    for box, score in zip(boxes, scores):
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 220, 0), 2)
        cv2.putText(sample, classes[names[i]], (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(sample, str(round(score, 2)), (box[0] + 180, box[1] - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (220, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.project, images), sample * 255)

        result_label_path = os.path.join(args.project, 'labels')

        isExists = os.path.exists(result_label_path)
        if not isExists:
            os.makedirs(result_label_path)  # 创建文件路径
        with open(os.path.join(result_label_path, os.path.splitext(images)[0] + '.txt'), 'a') as file:
            file.write("defect" + " " + str(round(score, 2)) + " " + " ".join([str(s) for s in box]) + '\n')

        i += 1
