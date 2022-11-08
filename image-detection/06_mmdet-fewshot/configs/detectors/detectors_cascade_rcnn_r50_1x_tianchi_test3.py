_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_eqlv2.py',
    '../_base_/datasets/coco_detection3.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale=dict(init_scale=512))
load_from = "z_detectors/epoch_60.pth"
model = dict(
    pretrained='checkpoints/resnet50-19c8e357.pth',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,  # resnet 50
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    # FPN->RFP
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            # resnet-50 imagenet预训练模型
            pretrained='checkpoints/resnet50-19c8e357.pth',
            style='pytorch'))
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

max_epochs = 20
num_last_epochs = 10
resume_from = None
interval = 2

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.01)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(interval=interval)

evaluation = dict(
    save_best='auto',
    interval=interval,
    metric='bbox')
# log_config = dict(interval=50)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])