import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model import LeNet
import pandas as pd
import os

PRED_PATH = 'E:\\Dataset\\mask\\test'
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')

net = LeNet()
net.fc3 = nn.Linear(84, len(classes))
net.load_state_dict(torch.load('weights\Lenet2.pth'))


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
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
        # y_pre.append(classes[int(predict)])
        y_pre['path'].append(img)
        y_pre['label'].append(classes[int(predict)])

# ----------------结果输出----------------
y_pre = pd.DataFrame(y_pre)
# result['label'] = y_pre
y_pre.to_csv('submit/result-lenet.csv', index=False)
# result.to_csv('submit/result-de2.csv', index=False)

# y_pre = {'path': [], 'label': []}
# for step, img in enumerate(file_name, start=0):
#     im = Image.open(os.path.join(PRED_PATH, img))
#     im = transform(im)  # [C, H, W]
#     im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(im.to(device))).cpu()
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()
#
#         y_pre['path'].append(img)
#         y_pre['label'].append(classes[int(predict_cla)])
#
# # ----------------结果输出----------------
# y_pre = pd.DataFrame(y_pre)
# y_pre.to_csv('submit/result-swin.csv', index=False)
