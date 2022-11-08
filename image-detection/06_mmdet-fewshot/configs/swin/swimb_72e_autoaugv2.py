_base_ = [
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=512.)
num_classes = 50
# model settings
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_base_patch4_window7_224.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            gc_context=True,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,  # True表示输出一个框，False表示输出num_classes个框
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # reg_decoded_bbox=True,
                # loss_bbox=dict(type='GIoULoss', loss_weight=13.0)
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # reg_decoded_bbox=True,
                # loss_bbox=dict(type='GIoULoss', loss_weight=13.0)
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # reg_decoded_bbox=True,
                # loss_bbox=dict(type='GIoULoss', loss_weight=13.0)
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)
            ),
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=5000,
            max_per_img=5000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            #  nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)))

dataset_type = 'TianchiDataset'
data_root = 'data/TianchiFilter/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Mosaic', img_scale=(1200, 1200), pad_val=114.0),
    dict(type='MixUp', img_scale=(1200, 1200), ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(
        type='Resize',
        # img_scale=[(1333, 600), (1333, 1000)],
        img_scale=[(1500, 800), (1500, 1200)],
        multiscale_mode='range',
        keep_ratio=True),

    dict(
        type='AutoAugment',
        autoaug_type="v2"),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=[
            # dict(type='HorizontalFlip', p=0.5),
            # dict(type='VerticalFlip', p=0.5),
            dict(type='RandomRotate90', p=0.5),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=1,
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=True,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1666, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1666, 1000), (1666, 1200), (1666, 1400), (1666, 1600)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        # type="ClassBalancedDataset",
        # oversample_thr=0.002,  # 重采样
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/wu.json',
            img_prefix=data_root + 'images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
            # pipeline=train_pipeline
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/wu.json',
        img_prefix=data_root + 'images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/val/' + 'annotations/instances_val2017.json',
        img_prefix='data/val/' + 'images/',
        pipeline=test_pipeline))

# optimizer_config = dict(grad_clip=None)
optimizer = dict(type='AdamW', lr=0.0000125 * 8, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

max_epochs = 72
# num_last_epochs = 5
# load_from = "/disk/sde/wzr/mmm/swimb/swim5.0/best_bbox_mAP_epoch_36.pth"
# resume_from = "/disk/sde/wzr/mmm/swimb/swimb2.0/best_bbox_mAP_epoch_34.pth"
interval = 2

# learning policy
# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=2,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.5)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[54, 66]

)

evaluation = dict(save_best='auto', interval=interval, metric='bbox')

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=interval)
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
optimizer_config = dict(grad_clip=None)
