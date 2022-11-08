import glob
import os.path
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data.dataloader as Dataloader
import qqdm
import numpy as np
import argparse

'''
    1. 参数设定
'''

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('--path', type=str, default='../../train/01/Images')
parser.add_argument('--Batchsize', type=int, default=2)
parser.add_argument('--fps', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

'''
    2. 定义数据集
'''


class MyDataset(nn.Module):
    def __init__(self, path, transforms):
        super(MyDataset, self).__init__()
        self.path = path
        self.transform = transforms
        self.imgs = glob.glob(os.path.join(self.path, '*.jpg'))

    def __getitem__(self, index):
        item = self.transform(Image.open(self.imgs[index % len(self.imgs)]))
        return item

    def __len__(self):
        return len(self.imgs)


'''
    3. 加载数据集
'''

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Resize([200, 200])])

dataset = MyDataset(args.path, transforms)
dataloader = Dataloader.DataLoader(dataset=dataset, batch_size=args.Batchsize, shuffle=True)

# print(len(dataset))


'''
    4. 定义 autoencoder
'''


class AutoencoderNet(nn.Module):

    def __init__(self):
        super(AutoencoderNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 1, 2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 5, 2, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 5, 1, 2), nn.ReLU()
        )

    def forward(self, X):
        output = self.layer(X)
        return output


'''
    5. 定义 网络、优化器、损失函数
'''

net = AutoencoderNet()
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
loss = nn.MSELoss()

'''
    6. 看看网络结构
'''
# X = torch.rand(size=(100, 3, 200, 200), dtype=torch.float32)
# X = X.cuda()
# for layer in net.layer:
#     X = layer(X)
#     # 输出对应层的shape
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)

'''
    7. 训练
'''
for epoch in range(args.epochs):
    processbar = qqdm.qqdm(dataloader)

    step = 0
    for i, data in enumerate(processbar):
        data = data.cuda()
        optimizer.zero_grad()
        image_out = net(data)
        print(image_out.shape, data.shape)
        l = loss(image_out, data)

        l.backward()
        optimizer.step()
        step += 1

        # processbar.set_info({
        #     'step': step,
        #     'epoch': epoch + 1
        # })

        print(l.item())
        if l.item() >= 0.007:
            print('第{}张图检测异常'.format((i+1)))
            img = image_out.cpu()
            # make_grid的作用是将若干幅图像拼成一幅图像
            img = torchvision.utils.make_grid(img, normalize=2)
            # 交换维度索引
            # plt.figure(figsize=(10,10))
            plt.imshow(img.permute(1, 2, 0))
            plt.show()

# torch.save(net, './model/detection.pkl')
