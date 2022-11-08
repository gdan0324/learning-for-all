import os
import argparse
import torch
from torchvision import transforms
import torch.optim as optim
from model import LeNet
import torch.nn as nn

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image


def create_data(path, batch_size, workers):
    # -------数据划分-------
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(path)
    # -------transform-------
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # -------dataset-------
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    # -------number of workers-------
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    print('Using {} dataloader workers'.format(nw))

    # -------dataloader-------
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             collate_fn=train_data_set.collate_fn)

    return train_loader, val_loader


def train(train_loader, val_loader, epochs, lr, save_path):
    """
        train_loader:训练数据集
        val_loader:验证数据集
        epochs:循环次数
        lr:学习率
        save_path:保存权重文件路径
    """
    # -------设备-------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # -------dataiterate-------
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    # -------定义网络-------
    net = LeNet()
    net.fc3 = nn.Linear(84, 3)
    # print(net)
    # -------损失函数-------
    loss_function = nn.CrossEntropyLoss()
    # -------优化器-------
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # -------开始训练-------
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    xxx = torch.max(outputs, dim=1)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='E:\\Dataset\\mask\\train', help='data path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for exp GPUs')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--learning-lr', type=int, default=0.001)
    parser.add_argument('--save-path', type=str, default=r'E:\save-github\deep-learning-all\image-classification'
                                                         r'\01_LeNet\weights\Lenet2.pth',
                        help='save weight path')
    opt = parser.parse_args()

    # -------数据准备-------
    train_loader, val_loader = create_data(opt.data, opt.batch_size, opt.workers)
    # -------开始训练-------
    train(train_loader, val_loader, opt.epochs, opt.learning_lr, opt.save_path)
