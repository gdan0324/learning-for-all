import torch
import torchvision.transforms as transforms
from PIL import Image
from model import AlexNet
import pandas as pd
import os
import json

PRED_PATH = 'E:\\Dataset\\mask\\test'

transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

# -------定义网络-------
net = AlexNet(num_classes=3).to(device)
net.load_state_dict(torch.load(r'E:\save-github\deep-learning-all\image-classification\02_AlexNet\weights\AlexNet.pth'))


def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


file_name = getFileList(PRED_PATH)
del (file_name[0])

net.eval()
y_pre = {'path': [], 'label': []}
for step, img in enumerate(file_name, start=0):
    im = Image.open(os.path.join(PRED_PATH, img))
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
    with torch.no_grad():
        # outputs = net(im)
        output = torch.squeeze(net(im.to(device))).cpu()
        # predict = torch.max(outputs, dim=1)[1].numpy()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # y_pre.append(classes[int(predict)])
        y_pre['path'].append(img)
        y_pre['label'].append(class_indict[str(predict_cla)])

# ----------------结果输出----------------
y_pre = pd.DataFrame(y_pre)
y_pre.to_csv('submit/result-alexnet.csv', index=False)
