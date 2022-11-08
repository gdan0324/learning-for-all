import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model import AlexNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')

    # -------定义网络-------
    net = AlexNet(num_classes=3)

    net.load_state_dict(torch.load(r'E:\save-github\deep-learning-all\image-classification\02_AlexNet\weights\AlexNet.pth'))

    im = Image.open('E:\\DL\\DL-objectclassification\\01_LeNet\\img.png')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
    # print(predict)


if __name__ == '__main__':
    main()
