import argparse
import os
import pandas as pd
import numpy as np
import cv2
import json

from sklearn import preprocessing

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset
from engine import train_one_epoch, evaluate
import warnings
import datetime

warnings.filterwarnings('ignore')


class officialDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        # print(image_id)
        # 读取pd的一条或多条记录
        records = self.df[self.df['image_id'] == image_id]
        # 根据id读入图片  cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道 cv2.IMREAD_GRAYSCALE：读入灰度图片 cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # 转换图片的颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 归一化图片
        image /= 255.0
        # open-cv提取的是 w*h*channel
        rows, cols = image.shape[:2]

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values

        # bbox的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 浅拷贝 转成32位浮点
        area = torch.as_tensor(area, dtype=torch.float32)

        # 取标签
        label = records['labels_num'].values
        # 转成整形
        labels = torch.as_tensor(label, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            # 放进 transforms
            sample = self.transforms(**sample)
            image = sample['image']

            # print(target['boxes'])
            # torch.stack 打包 .permute(1,0) 相当于转置
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

            return image, target

    def __len__(self) -> int:
        # 返回数据集的数量
        return self.image_ids.shape[0]


# 把每个异常点都抽出来
class thirdDataset(object):
    def __init__(self, json_file, imgs_dir):
        self.json_file = json_file
        self.imgs_dir = imgs_dir
        # 读取json文件
        with open(os.path.join(self.json_file), 'r') as load_f:
            load_dict = json.load(load_f)
        self.labels = []
        self.boxes = []
        self.image_id = []
        self.img_path = []
        for item in load_dict:
            self.labels.extend(item['labels'])
            self.boxes.extend(item['boxes'])
            # 一张图片里面有多少个[item['name']]都附加进去
            self.image_id.extend([item['name']] * len(item['boxes']))
            self.img_path.extend([os.path.join(self.imgs_dir, item['name'] + '.jpg')] * len(item['boxes']))

    def to_df(self):
        a = {"image_id": self.image_id,
             "labels": self.labels,
             "boxes": self.boxes,
             "img_path": self.img_path}
        df = pd.DataFrame.from_dict(a, orient='index')
        # 转置
        df = df.transpose()
        return df


def train(opt):
    BASE_PATH = opt.source
    IMG_PATH = os.path.join(BASE_PATH, "abnormImages")
    JSON_PATH = os.path.join(BASE_PATH, "TRAIN_objects.json")

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir = opt.output_dir + "/weight{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # make dictionary for class objects so we can call objects by their keys.
    num_classes = opt.num_classes + 1  # 多一个背景类

    batch_size = opt.batch_size
    num_epochs = opt.epochs

    dataset = thirdDataset(JSON_PATH, IMG_PATH)
    df = dataset.to_df()

    # classes need to be in int form so we use LabelEncoder for this task
    # 将标签转换成1、2、3  0--背景类
    enc = preprocessing.LabelEncoder()
    df['labels_num'] = enc.fit_transform(df['labels'])
    # 将标签值全部加1
    df['labels_num'] = np.stack(df['labels_num'][i] + 1 for i in range(len(df['labels_num'])))

    # bounding box coordinates point need to be in separate columns
    df['xmin'] = -1
    df['ymin'] = -1
    df['xmax'] = -1
    df['ymax'] = -1

    # 将bbox扩展开来
    df[['xmin', 'ymin', 'xmax', 'ymax']] = np.stack(df['boxes'][i] for i in range(len(df['boxes'])))

    df.drop(columns=['boxes'], inplace=True)
    df['xmin'] = df['xmin'].astype(np.float)
    df['ymin'] = df['ymin'].astype(np.float)
    df['xmax'] = df['xmax'].astype(np.float)
    df['ymax'] = df['ymax'].astype(np.float)

    # 划分训练集和验证集
    silce = int(0.2 * len(df['image_id'].unique()))
    image_ids = df['image_id'].unique()
    valid_ids = image_ids[-silce:]
    train_ids = image_ids[:-silce]

    # 把验证集、训练集的数据拿出来
    valid_df = df[df['image_id'].isin(valid_ids)]
    train_df = df[df['image_id'].isin(train_ids)]

    # 训练集的数据增强
    def get_transform_train():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    # 测试集转tensor
    def get_transform_valid():
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    train_dataset = officialDataset(train_df, IMG_PATH, get_transform_train())
    valid_dataset = officialDataset(valid_df, IMG_PATH, get_transform_valid())

    def collate_fn(batch):
        return tuple(zip(*batch))

    # DataLoader
    # collate_fn 合并一个样本列表，形成一个张量(s)的小批。 当从映射风格的数据集中使用批处理加载时使用。

    # -------number of workers-------
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, opt.workers])
    print('Using {} dataloader workers'.format(nw))

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 原本是1024类的改为4类的输出

    # get number of input features for the classifier
    # 获取分类器的输入特征的数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(model.roi_heads.box_predictor)

    model.to(device)

    # 将需要更新的权重更新
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # 训练一个时期，每10次迭代打印一次
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        print_freq=1000 // batch_size)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_info = evaluate(model, valid_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, os.path.join(output_dir, "resNetFpn-model-{}.pth".format(epoch)))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == '__main__':
    # -------------------------参数-------------------------------
    parser = argparse.ArgumentParser(description="train object")
    parser.add_argument('--weights', nargs='+', type=str, default='models/faster_rcnn_state11.pth',
                        help='model.pt path(s)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--source', type=str, default=r'../../../../train/01/')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for exp GPUs')
    parser.add_argument('--lr', type=int, default=0.005)
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=3, type=int, help='num_classes')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 权重保存地址
    parser.add_argument('--output-dir', default='weights', help='path where to save')

    opt = parser.parse_args()

    # -------开始训练-------
    train(opt)
