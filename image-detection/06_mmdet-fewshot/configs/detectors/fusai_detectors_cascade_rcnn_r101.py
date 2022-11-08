_base_ = [
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=512.)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='DetectoRS_ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        output_img=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnest50_d2-7497a55b.pth')
    ),
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
            type='DetectoRS_ResNeSt',
            stem_channels=64,
            depth=50,
            radix=2,
            reduction_factor=4,
            avg_down_stride=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=norm_cfg,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            norm_eval=True,
            output_img=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnest50_d2-7497a55b.pth')),
    ),
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
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
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
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     loss_weight=1.0),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                loss_cls=dict(type='EQLv2'),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
            )
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                gpu_assign_thr=1,
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
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    gpu_assign_thr=1,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',  # OHEMSampler
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
                    gpu_assign_thr=1,
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
                    gpu_assign_thr=1,
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
            score_thr=0.01,  # 0.0001
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            # nms=dict(type='nms', iou_threshold=0.99),
            max_per_img=300)))

# dataset setting
dataset_type = 'CocoDataset'
data_root = 'data/fusai/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MixUp', img_scale=(1200, 1200), ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(
        type='Resize',
        img_scale=[(2048, 800), (2048, 1600)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
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
        type='AutoAugment',
        autoaug_type="v2"),
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
        img_scale=(2048, 1600),
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
            (2048, 1000), (2048, 1200), (2048, 1400), (2048, 1600)],
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
            ann_file=data_root + 'annotations/instances_train2017.json',
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
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/fusai_val/' + 'annotations/instances_val2017.json',
        img_prefix='data/fusai_val/' + 'images/',
        pipeline=test_pipeline))

optimizer = dict(type='AdamW', lr=0.0000125 * 8, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

max_epochs = 72
# num_last_epochs = 3
interval = 4
load_from = "fusai/detectors/resnest101_36e_chusai/epoch_36.pth"
# resume_from = "/disk/sde/wzr/mmm/convnext/convnext2.0/best_bbox_mAP_epoch_132.pth"

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[54, 66])

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
