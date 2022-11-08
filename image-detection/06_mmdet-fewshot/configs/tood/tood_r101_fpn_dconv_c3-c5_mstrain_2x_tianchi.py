_base_ = './tood_r50_fpn_1x_coco.py'

fp16 = dict(loss_scale=dict(init_scale=512))
# load_from = "checkpoints/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth"
model = dict(
    backbone=dict(
        depth=101,
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(num_dcn=2))

# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 1280)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
max_epochs = 200
num_last_epochs = 15
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

custom_hooks = [dict(type='SetEpochInfoHook')]

checkpoint_config = dict(
    interval=4,
    )

evaluation = dict(
    save_best='auto',
    interval=interval,
    metric='bbox')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

