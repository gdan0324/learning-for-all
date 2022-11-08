import os
import datetime
import torch
import torchvision

import train_utils.transforms as transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from my_dataset import VOCDataSet
import argparse
from train_utils import train_eval_utils as utils
from backbone import BackboneWithFPN, LastLevelMaxPool


def create_data(path, batch_size, workers):
    # -------transform-------
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # check voc root
    VOC_root = path  # VOCdevkit
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # -------dataset-------
    # VOCdevkit -> VOC2007 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2007", data_transform["train"], "train.txt")
    # VOCdevkit -> VOC2007 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2007", data_transform["val"], "val.txt")
    print(len(train_dataset))

    # -------number of workers-------
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    print('Using %g dataloader workers' % nw)

    # -------dataloader-------
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # aspect_ratio_group_factor = 3
    # amp = False  # 是否使用混合精度训练，需要GPU支持
    #
    # train_sampler = None
    #
    # # 是否按图片相似高宽比采样图片组成batch
    # # 使用的话能够减小训练时所需GPU显存，默认使用
    # if aspect_ratio_group_factor >= 0:
    #     train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #     # 统计所有图像高宽比例在bins区间中的位置索引
    #     group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
    #     # 每个batch图片从同一高宽比例区间中取
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    #
    # # -------dataloader-------
    # # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    # if train_sampler:
    #     # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    #     train_data_loader = torch.utils.data.DataLoader(train_dataset,
    #                                                     batch_sampler=train_batch_sampler,
    #                                                     pin_memory=True,
    #                                                     num_workers=nw,
    #                                                     collate_fn=train_dataset.collate_fn)
    # else:
    #     train_data_loader = torch.utils.data.DataLoader(train_dataset,
    #                                                     batch_size=batch_size,
    #                                                     shuffle=True,
    #                                                     pin_memory=True,
    #                                                     num_workers=nw,
    #                                                     collate_fn=train_dataset.collate_fn)

    return train_data_loader, val_data_loader


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512

    # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    backbone = MobileNetV2(weights_path="backbone/mobilenet_v2-pre.pth").features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    # 下采样16倍
    # vgg16
    # backbone = torchvision.models.vgg16_bn(pretrained=False)
    # # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    # backbone.out_channels = 512

    # resnet50 backbone
    # backbone = torchvision.models.resnet50(pretrained=False)
    # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"layer3": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    # backbone.out_channels = 1024

    # EfficientNetB0
    # backbone = torchvision.models.efficientnet_b0(pretrained=False)
    # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"features.5": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    # backbone.out_channels = 112

    # --- mobilenet_v3_large fpn backbone --- #
    # backbone = torchvision.models.mobilenet_v3_large(pretrained=False)
    # # print(backbone)
    # # 抽取多个特征层
    # return_layers = {"features.6": "0",  # stride 8
    #                  "features.12": "1",  # stride 16
    #                  "features.16": "2"}  # stride 32
    # in_channels_list = [40, 112, 960]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    # --- efficientnet_b0 fpn backbone --- #
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # return_layers = {"features.3": "0",  # stride 8
    #                  "features.4": "1",  # stride 16
    #                  "features.8": "2"}  # stride 32
    # # 提供给fpn的每个特征层channel
    # in_channels_list = [40, 80, 1280]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # # img = torch.randn(1, 3, 224, 224)
    # # outputs = new_backbone(img)
    # # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    # backbone_with_fpn = BackboneWithFPN(new_backbone,
    #                                     return_layers=return_layers,
    #                                     in_channels_list=in_channels_list,
    #                                     out_channels=256,
    #                                     extra_blocks=LastLevelMaxPool(),    # maxpool只用于rpn，不用于fast rcnn部分
    #                                     re_getter=False  # 是否重构
    #                                     )
    # anchor_sizes = ((64,), (128,), (256,), (512,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
    #                                     aspect_ratios=aspect_ratios)
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
    #                                                 output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
    #                                                 sampling_ratio=2)  # 采样率
    # model = FasterRCNN(backbone=backbone_with_fpn,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # 每个滑动窗口预测 5 乘以 3 个anchor
    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def train(train_data_loader, val_data_loader, args):
    """
        train_loader:训练数据集
        val_loader:验证数据集
        args:参数
    """
    # -------设备-------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # -------dataiterate-------
    train_data_iter = iter(train_data_loader)
    train_image, train_target = train_data_iter.next()

    # -------定义网络-------
    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    amp = False  # 是否使用混合精度训练，需要GPU支持
    scaler = torch.cuda.amp.GradScaler() if amp else None

    # -------损失函数-------
    train_loss = []
    learning_rate = []
    val_map = []

    # -------优化器-------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    # -------开始训练-------
    init_epochs = 5
    for epoch in range(init_epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

    save_weight_path = os.path.join(args.save_weight,
                                    "weight{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)

    # torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 冻结backbone部分底层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_lr,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    num_epochs = args.epochs
    for epoch in range(init_epochs, num_epochs + init_epochs, 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        # 仅保存最后5个epoch的权重
        if epoch in range(num_epochs + init_epochs)[-5:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, os.path.join(save_weight_path, "mobile-model-{}.pth".format(epoch)))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from train_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from train_utils.plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='data path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for exp GPUs')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--learning-lr', type=int, default=0.005)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--save-weight', type=str,
                        default=r'E:\save-github\deep-learning-all\object-detection\01_faster_rcnn\save_weights',
                        help='save weight path')
    parser.add_argument('--weights', type=str,
                        default=r'checkpoints/mobilenet_v2-pre.pth',
                        help='initial weights path')
    opt = parser.parse_args()
    # -------数据准备-------
    train_loader, val_loader = create_data(opt.data, opt.batch_size, opt.workers)
    # -------开始训练-------
    train(train_loader, val_loader, opt)
    # main()
