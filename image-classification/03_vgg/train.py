import json
import os
import argparse
import time

import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from model import vgg
import torch.nn as nn
from tqdm import tqdm
import sys
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def create_data(path, batch_size, workers):
    # -------数据划分-------
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(path)

    # -------transform-------
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   # transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # -------dataset-------
    train_data_set = datasets.ImageFolder(root=path + '/train', transform=data_transform["train"])
    val_data_set = datasets.ImageFolder(root=path + '/val', transform=data_transform["val"])

    # ------- 写json - ------
    classes = train_data_set.class_to_idx
    classes = dict((value, key) for key, value in classes.items())
    json_str = json.dumps(classes, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

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
        epochs:循环次数
        lr:学习率
        num_classes:类别数
        save_path:保存权重文件路径
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
    model_name = "vgg16"
    net = vgg(model_name, num_classes=args.num_classes, init_weights=True)
    net.to(device)
    # # -------加载预训练权重-------
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "classifier" in k:
    #             del weights_dict[k]
    #     print(net.load_state_dict(weights_dict, strict=False))

    # -------损失函数-------
    loss_function = nn.CrossEntropyLoss()
    # -------优化器-------
    optimizer = optim.Adam(net.parameters(), lr=args.learning_lr)

    # -------数量--------
    val_num = len(val_loader)
    train_num = len(train_loader)

    # -------tensorboard--------
    # tb_writer = SummaryWriter()

    best_acc = 0.0

    # -------开始训练-------
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        start_time = time.perf_counter()
        # train_bar = tqdm(train_loader, file=sys.stdout)
        # batch用enumerate
        for step, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        # 打印训练所需时间
        print(time.perf_counter() - start_time)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            # val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), args.save_path)
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
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for exp GPUs')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--learning-lr', type=int, default=0.0002)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--save-path', type=str,
                        default=r'E:\save-github\deep-learning-all\image-classification\03_vgg\weights\vggnet2.pth',
                        help='save weight path')
    # parser.add_argument('--weights', type=str, default=r'E:\save-github\deep-learning-all\image-classification\03_vgg\checkpoints\vgg16-397923af.pth',
    #                     help='initial weights path')
    opt = parser.parse_args()

    # -------数据准备-------
    train_loader, val_loader = create_data(opt.data, opt.batch_size, opt.workers)
    # -------开始训练-------
    # train(train_loader, val_loader, opt.epochs, opt.learning_lr, opt.num_classes, opt.save_path)
    train(train_loader, val_loader, opt)
