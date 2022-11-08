_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_eqlv2.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale=dict(init_scale=512))

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        pretrained="checkpoints/resnet50-19c8e357.pth"),
)


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.004,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

max_epochs = 200
num_last_epochs = 15
resume_from = "cascade/cascade2.0/best_bbox_mAP_epoch_170.pth"
interval = 2

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.01)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(interval=interval*2)

evaluation = dict(
    save_best='auto',
    interval=interval,
    metric='bbox')
# log_config = dict(interval=50)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
