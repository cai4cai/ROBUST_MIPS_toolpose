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

NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
max_epochs = 600 
val_interval = 5 
train_batch_size = 32
val_batch_size = 8

train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[550, 570],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = {
    'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=152,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet152'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=NUM_KEYPOINTS,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'dataset/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=2,
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
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='cocoformat_val.json',
        bbox_file=data_root+'converted_detections_val.json',
        data_prefix=dict(img='val/img/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='cocoformat_test.json',
        bbox_file=data_root+'converted_detections_test.json',
        data_prefix=dict(img='testing/img/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

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
