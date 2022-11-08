import os
import json
import time

import torch
from PIL import Image
import matplotlib.pyplot as plt

import train_utils.transforms as transforms
from backbone import SSD300, Backbone


def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def getFileList(path):
    for root, dirs, files in os.walk(path):
        return files


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=2)

    # load train weights
    weights_path = r"E:\save-github\deep-learning-all\object-detection\02_ssd\save_weights\weight20220903-102345\ssd300-model-29.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # read class_indict
    # label_json_path = 'data/pascal_voc_fsod.json'
    # assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    # with open(label_json_path, 'r') as f:
    #     class_dict = json.load(f)

    # category_index = {str(v): str(k) for k, v in class_dict.items()}

    RESULT_PATH = r'E:\save-github\deep-learning-all\object-detection\02_ssd\save_weights\weight20220903-102345\detect-results'
    # ------------------------创建结果文件夹----------------------------------
    isExists = os.path.exists(RESULT_PATH)
    if not isExists:
        os.makedirs(RESULT_PATH)  # 创建文件路径

    # load image
    PRED_PATH = r'C:\Users\96212\Desktop\abnorm\test\images'
    file_name = getFileList(PRED_PATH)
    for step, img in enumerate(file_name, start=0):
        original_img = Image.open(os.path.join(PRED_PATH, img))
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.Resize(),
                                             transforms.ToTensor(),
                                             transforms.Normalization()])
        # data_transform = transforms.Compose([transforms.ToTensor()])
        img, _ = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            # img_height, img_width = img.shape[-2:]
            # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            init_img = torch.zeros((1, 3, 300, 300), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                with open(os.path.join(RESULT_PATH, str(step + 1).rjust(3, '0') + '.txt'), 'a') as file:
                    pass
            else:
                for box, score in zip(predict_boxes, predict_scores):
                    with open(os.path.join(RESULT_PATH, str(step + 1).rjust(3, '0') + '.txt'), 'a') as file:
                        file.write(
                            "defect" + " " + str(round(score, 6)) + " " + " ".join([str(int(s)) for s in box]) + '\n')


if __name__ == '__main__':
    main()
