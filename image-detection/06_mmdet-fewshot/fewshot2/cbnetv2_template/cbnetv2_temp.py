checkpoint_config = dict(interval=2)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
fp16 = dict(loss_scale=512.0)
model = dict(
    type='CascadeRCNNTemplate',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'
    ),
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=128,
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
        out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='SiameseFPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            scales=[8],
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
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0))
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
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
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=300)))
dataset_type = 'FewShotTestDataset'
data_root = 'data/fewshot2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFileWithTemplate',
        template_path='data/fewshot2//train/Template'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ResizeWithTemplate',
        img_scale=(300, 300),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCenterCropPadWithTemplate',
        crop_size=(200, 200),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        to_rgb=False,
        test_pad_mode=None),
    dict(
        type='RandomFlipWithTemplate',
        direction=['horizontal', 'vertical'],
        flip_ratio=0.5),
    dict(type='AutoAugmentWithTemplate', autoaug_type='v1'),
    dict(
        type='NormalizeWithTemplate',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadWithTemplate', size_divisor=32),
    dict(type='DefaultFormatBundle_Template'),
    dict(
        type='Collect', keys=['img', 'template_img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFileWithTemplate',
        template_path='data/fewshot2//train/Template'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=True,
        transforms=[
            dict(type='ResizeWithTemplate', keep_ratio=True),
            dict(
                type='RandomFlipWithTemplate',
                direction=['horizontal', 'vertical']),
            dict(
                type='NormalizeWithTemplate',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadWithTemplate', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'template_img']),
            dict(type='Collect', keys=['img', 'template_img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='FewShotTestDataset',
        ann_file='data/fewshot2/train_annotation.json',
        img_prefix='data/fewshot2/train/JPEGImages/',
        filter_empty_gt=False,
        pipeline=[
            dict(
                type='LoadImageFromFileWithTemplate',
                template_path='data/fewshot2//train/Template'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='ResizeWithTemplate',
                img_scale=(300, 300),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCenterCropPadWithTemplate',
                crop_size=(200, 200),
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                to_rgb=False,
                test_pad_mode=None),
            dict(
                type='RandomFlipWithTemplate',
                direction=['horizontal', 'vertical'],
                flip_ratio=0.5),
            dict(type='AutoAugmentWithTemplate', autoaug_type='v1'),
            dict(
                type='NormalizeWithTemplate',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadWithTemplate', size_divisor=32),
            dict(type='DefaultFormatBundle_Template'),
            dict(
                type='Collect',
                keys=['img', 'template_img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='FewShotTestDataset',
        ann_file='data/fewshot2/train_annotation.json',
        img_prefix='data/fewshot2/train/JPEGImages/',
        pipeline=[
            dict(
                type='LoadImageFromFileWithTemplate',
                template_path='data/fewshot2//train/Template'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=True,
                transforms=[
                    dict(type='ResizeWithTemplate', keep_ratio=True),
                    dict(
                        type='RandomFlipWithTemplate',
                        direction=['horizontal', 'vertical']),
                    dict(
                        type='NormalizeWithTemplate',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='PadWithTemplate', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 'template_img']),
                    dict(type='Collect', keys=['img', 'template_img'])
                ])
        ]),
    test=dict(
        type='FewShotTestDataset',
        ann_file='data/fewshot_test2.json',
        img_prefix='data/fewshot2/test/JPEGImages/',
        pipeline=[
            dict(
                type='LoadImageFromFileWithTemplate',
                template_path='data/fewshot2//train/Template'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=True,
                transforms=[
                    dict(type='ResizeWithTemplate', keep_ratio=True),
                    dict(
                        type='RandomFlipWithTemplate',
                        direction=['horizontal', 'vertical']),
                    dict(
                        type='NormalizeWithTemplate',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='PadWithTemplate', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 'template_img']),
                    dict(type='Collect', keys=['img', 'template_img'])
                ])
        ]))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
max_epochs = 1
num_last_epochs = 6
interval = 2
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=6,
    min_lr_ratio=0.1)
evaluation = dict(interval=1, metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = 'fewshot2/cbnetv2_template/'
auto_resume = False
gpu_ids = [0]
