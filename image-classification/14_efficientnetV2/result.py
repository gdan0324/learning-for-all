import torch
import torchvision.transforms as transforms
from PIL import Image
from model import efficientnetv2_s as create_model
import pandas as pd
import os
import json

PRED_PATH = 'E:\\Dataset\\mask\\test'
# 读取指定文件夹下所有jpg图像路径
img_path_list = [os.path.join(PRED_PATH, i) for i in os.listdir(PRED_PATH)]

img_size = {"s": [300, 384],  # train_size, val_size
            "m": [384, 480],
            "l": [384, 480]}
num_model = "s"

data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

# -------定义网络-------
# net = resnet34(num_classes=3).to(device)
net = create_model(num_classes=3).to(device)
# load model weights
model_weight_path = "weights/model-25.pth"
assert os.path.exists(model_weight_path), "file: '{}' dose not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
net.eval()
batch_size = 1  # 每次预测时将多少张图片打包成一个batch

# def getFileList(path):
#     for root, dirs, files in os.walk(path):
#         return files
#
#
# file_name = getFileList(PRED_PATH)
# del (file_name[0])

y_pre = {'path': [], 'label': []}
with torch.no_grad():
    for ids in range(0, len(img_path_list) // batch_size):

        img_list = []
        img_name = []
        for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            img = Image.open(img_path)
            img = data_transform(img)
            img_list.append(img)
            y_pre['path'].append(os.path.basename(img_path))

        # batch img
        # 将img_list列表中的所有图像打包成一个batch
        batch_img = torch.stack(img_list, dim=0)
        # predict class
        output = net(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        # predict_cla = torch.argmax(predict).numpy()

        probs, predict_cla = torch.max(predict, dim=1)

        for idx, (pro, cla) in enumerate(zip(probs, predict_cla)):
            y_pre['label'].append(class_indict[str(cla.numpy())])
            # print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
            #                                                  class_indict[str(cla.numpy())],
            #                                                  pro.numpy()))

# ----------------结果输出----------------
y_pre = pd.DataFrame(y_pre)
y_pre.to_csv('submit/result-efficientnet.csv', index=False)
