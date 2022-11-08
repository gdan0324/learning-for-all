import json
import os
import argparse
import time

import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from model_v2 import MobileNetV2
import torch.nn as nn
from tqdm import tqdm
import sys
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
from model_v3 import mobilenet_v3_large
val_num = 0
train_num = 0


def create_data(path, batch_size, workers):
    # -------数据划分-------
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(path)

    # -------transform-------
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # -------dataset-------
    # train_data_set = datasets.ImageFolder(root=path + '/train', transform=data_transform["train"])
    # val_data_set = datasets.ImageFolder(root=path + '/val', transform=data_transform["val"])
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    # -------更改全局num-------
    global train_num, val_num
    train_num = len(train_data_set)
    val_num = len(val_data_set)

    # ------- 写json - ------
    # classes = train_data_set.class_to_idx
    # classes = dict((value, key) for key, value in classes.items())
    # json_str = json.dumps(classes, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    # -------number of workers-------
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    print('Using {} dataloader workers'.format(nw))

    # -------dataloader-------
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    return train_loader, val_loader


def train(train_loader, val_loader, args):
    """
        train_loader:训练数据集
        val_loader:验证数据集
        args:参数
    """
    # -------设备-------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # -------dataiterate-------
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    # # -------可以做一些数据可视化操作，主要是验证我们读入的数据是否正确-------
    # val_image = utils.make_grid(val_image)
    # val_image = val_image / 2 + 0.5
    # val_image = val_image.numpy()
    # plt.imshow(np.transpose(val_image, (1, 2, 0)))
    # plt.show()

    # -------定义网络-------
    net = mobilenet_v3_large(num_classes=args.num_classes)

    # # -------加载预训练权重-------
    model_weight_path = args.weights

    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # pre_dict = {k: v for k, v in pre_weights.items() if "Classifier" not in k}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    # for param in net.features.parameters():
    #     param.requires_grad = False
    net.to(device)

    # -------损失函数-------
    loss_function = nn.CrossEntropyLoss()
    # -------优化器-------
    optimizer = optim.Adam(net.parameters(), lr=args.learning_lr)

    # # -------数量--------
    # val_num = len(val_loader)
    # train_num = len(train_loader)

    # -------tensorboard--------
    # tb_writer = SummaryWriter()

    best_acc = 0.0

    # -------开始训练-------
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        # batch用enumerate
        for step, data in enumerate(train_bar):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            logits = net(inputs.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)
        val_accurate = acc / val_num

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(args.save_path, 'MobileNetv3.pth'))

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_num, val_accurate))

        # tensorboard
        # tags = ["train_loss", "val_accurate", "learning_rate"]
        # tb_writer.add_scalar(tags[0], running_loss / train_num, epoch)
        # tb_writer.add_scalar(tags[1], val_accurate, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    print('Finished Training')


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='E:\\Dataset\\mask\\train', help='data path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for exp GPUs')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--learning-lr', type=int, default=0.0002)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--save-path', type=str,
                        default=r'E:\save-github\deep-learning-all\image-classification\06_MobileNet\weights',
                        help='save weight path')
    parser.add_argument('--weights', type=str,
                        default=r'checkpoints/mobilenet_v3_pre.pth',
                        help='initial weights path')
    opt = parser.parse_args()

    # -------数据准备-------
    train_loader, val_loader = create_data(opt.data, opt.batch_size, opt.workers)
    # -------开始训练-------
    # train(train_loader, val_loader, opt.epochs, opt.learning_lr, opt.num_classes, opt.save_path)
    train(train_loader, val_loader, opt)
