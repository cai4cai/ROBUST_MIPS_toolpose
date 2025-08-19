_base_ = ['../../mmpose/configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'evaluation.metric.swap_coco_metric'],
    allow_failed_imports=False) 

dataset_info = {
    'dataset_name':'ROBUST-MIPS',
    'classes':'SurgicalTool',
    'paper_info':{
        'author':'Zhe Han, Charlie Budd, Gongyu Zhang, Huanyu Tian, Christos Bergeles, Tom Vercauteren',
        'title':'ROBUST-MIPS: A Combined Skeletal Pose Representation and Instance Segmentation Dataset for Laparoscopic Surgical Instruments',
        'year':'2025'
    },
    'keypoint_info':{
        0:{'name':'entry','id':0,'color':[255,0,0],'type': '','swap': ''},
        1:{'name':'hinge','id':1,'color':[0,255,0],'type': '','swap': ''},
        2:{'name':'tip1','id':2,'color':[0,0,255],'type': '','swap': ''},
        3:{'name':'tip2','id':3,'color':[255,255,0],'type': '','swap': ''}
    },
    'skeleton_info': {
        0: {'link':('entry','hinge'),'id': 0,'color': [255,0,255]}, 
        1: {'link':('hinge','tip1'),'id': 1,'color': [0,255,255]}, 
        2: {'link':('hinge','tip2'),'id': 2,'color': [255,165,0]} 
    },
    'joint_weights':[
        1., 1., 1., 1. 
    ],
    'sigmas':[
        0.107, 0.107, 0.107, 0.107 
    ]
}

# number of keypoints
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])

# parameters
max_epochs = 600 
val_interval = 5 
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 32 
val_batch_size = 16

stage2_num_epochs = 30
base_lr = 4e-3

randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'dataset/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='cocoformat_train.json',
        data_prefix=dict(img='training/img/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='cocoformat_val.json',
        bbox_file=data_root+'converted_detections_0123_val.json',
        data_prefix=dict(img='val/img/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='cocoformat_test.json',
        bbox_file=data_root+'converted_detections_0123_test.json',
        data_prefix=dict(img='testing/img/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# hooks
default_hooks = {
    'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'cocoformat_val.json')
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='SwapCocoMetric',
    ann_file=data_root + 'cocoformat_test.json')

vis_backends = [dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]
visualizer = dict(
    vis_backends=vis_backends,
    name='visualizer')
