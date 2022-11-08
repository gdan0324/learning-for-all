import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import sys

from model import swin_tiny_patch4_window7_224 as create_model
import pandas as pd
import os

PRED_PATH = r'E:\Dataset\mask\test'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 224
transform = transforms.Compose(
    [transforms.Resize(int(img_size * 1.14)),
     transforms.CenterCrop(img_size),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')

# create model
model = create_model(num_classes=3).to(device)
# load model weights
model_weight_path = "./weights/model-21.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


file_name = getFileList(PRED_PATH)
del (file_name[0])

y_pre = {'path': [], 'label': []}
for step, img in enumerate(file_name, start=0):
    im = Image.open(os.path.join(PRED_PATH, img))
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(im.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        y_pre['path'].append(img)
        y_pre['label'].append(classes[int(predict_cla)])

# ----------------结果输出----------------
y_pre = pd.DataFrame(y_pre)
y_pre.to_csv('submit/result-swin.csv', index=False)
