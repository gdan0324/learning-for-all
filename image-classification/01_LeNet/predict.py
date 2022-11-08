import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('mask_weared_incorrect', 'with_mask', 'without_mask')

    net = LeNet()
    net.fc3 = nn.Linear(84, len(classes))
    net.load_state_dict(torch.load(r'E:\save-github\deep-learning-all\image-classification\01_LeNet\weights\Lenet2.pth'))

    im = Image.open('DL-objectclassification\LeNet\img.png')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
    # print(predict)


if __name__ == '__main__':
    main()
