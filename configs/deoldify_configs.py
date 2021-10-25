# custom_imports = dict(imports=['my_pipelines'], allow_failed_imports=False)
import sys
sys.path.append('/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/configs')

custom_imports = dict(
    imports=['deoldify', 'resnet_backbone', 'channels_from_one_to_three'],
    allow_failed_imports=False)

model = dict(
    type='DeOldify',
    encoder=dict(
        type='ColorizationResNet',
        num_layers=101,
        pretrained=None),
    n_classes=3,
    blur=True,
    blur_final=True,
    self_attention=True,
    y_range=(-3.0, 3.0),
    last_cross=True,
    bottle=False,
    nf_factor=2
    )

train_cfg = dict()
test_cfg = dict()


dataset_type = ''


train_pipeline = []
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt_img',
        flag='grayscale',
        backend='pillow'
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=(160, 160),
        keep_ratio=False,
        backend='pillow'
    ),
    dict(type='RescaleToZeroOne', keys=['gt_img']),
    dict(
        type='ChannelsFromOneToThree',
        keys=['gt_img'],
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[0.4850, 0.4560, 0.4060],
        std=[0.2290, 0.2240, 0.2250],
        to_rgb=False),
    dict(
        type='Collect',
        keys=['gt_img'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor',
         keys=['gt_img']),
]


