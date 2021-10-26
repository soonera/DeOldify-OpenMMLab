# custom_imports = dict(imports=['my_pipelines'], allow_failed_imports=False)
import sys
# sys.path.append('/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/configs')
sys.path.append('/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/models')
sys.path.append('/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/apis')
sys.path.append('/home/SENSETIME/renqin/PycharmProjects/DeOldify-demo/datasets')


custom_imports = dict(
    imports=['deoldify', 'resnet_backbone', 'channels_from_one_to_three'],
    allow_failed_imports=False)

model = dict(
    type='DynamicUnetDeep',
    encoder=dict(
        type='ColorizationResNet',
        num_layers=34,
        pretrained=None),
    nf_factor=1.5)

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
        # scale=(35 * 16, 35 * 16),
        scale=(25 * 16, 25 * 16),
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


