_base_ = [
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=512.)
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        pretrained="checkpoints/convnext_base_1k_224.pth"
    ),
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
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],  # 增加到7个
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,  # 3->4
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            gc_context=True,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,  # True表示输出一个框，False表示输出num_classes个框
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # norm_cfg=dict(type='BN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # loss_cls=dict(
                #     type='SeesawLoss',
                #     p=0.8,
                #     q=2.0,
                #     num_classes=50,
                #     loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # loss_cls=dict(
                #     type='SeesawLoss',
                #     p=0.8,
                #     q=2.0,
                #     num_classes=50,
                #     loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # loss_cls=dict(
                #     type='SeesawLoss',
                #     p=0.8,
                #     q=2.0,
                #     num_classes=50,
                #     loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,  # 0.7->0.5
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
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,  # 0.5->0.3
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    # type='OHEMSampler',
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
                    pos_iou_thr=0.6,  # 0.6->0.4
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
                    pos_iou_thr=0.7,  # 0.7->0.5
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
            nms_across_levels=False,
            nms_pre=5000,
            nms_post=5000,
            max_per_img=5000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            # nms=dict(type='nms', iou_threshold=0.99),
            max_per_img=300)))

dataset_type = 'CocoDataset'
data_root = 'data/TianchiFilter/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ChannelShuffle', p=0.1)
]

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MixUp', img_scale=(1200, 1200), ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(
        type='Resize',
        img_scale=[(1666, 800), (1666, 1200)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0)
                ],
                p=0.1),
            dict(type='ChannelShuffle', p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=True,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1666, 1200),
        flip=False,
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[
            (1666, 1000), (1666, 1200), (1666, 1400), (1666, 1600)],
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
        type="MultiImageMixDataset",
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
        ann_file=data_root + 'annotations/wu_val.json',
        img_prefix=data_root + 'images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/val/' + 'annotations/instances_val2017.json',
        img_prefix='data/val/' + 'images/',
        pipeline=test_pipeline))

optimizer = dict(type='AdamW', lr=0.0000125 * 4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

max_epochs = 36
# num_last_epochs = 3
interval = 2
# load_from = "/disk/sde/wzr/mmm/checkpoints/cascade_mask_rcnn_convnext_base_1k_3x.pth"
# resume_from = "/disk/sde/wzr/mmm/convnext/convnext2.0/best_bbox_mAP_epoch_132.pth"

# learning policy
# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=3,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.1)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[27, 33])

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
