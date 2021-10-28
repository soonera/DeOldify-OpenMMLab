custom_imports = dict(
    imports=['models', 'apis'],
    allow_failed_imports=False)

model = dict(
    type='DeOldify',
    encoder=dict(
        type='ColorizationResNet',
        num_layers=34,
        pretrained=None,
        out_layers=[2, 5, 6, 7]),
    mid_layers=dict(
        # channel_factors=[2, 1],
        type='MidConvLayer',
        norm_type="NormSpectral",
        ni=512),
    decoder=dict(
        type='UnetDeepDecoder',
        self_attention=True,
        x_in_c_list=[64, 64, 128, 256],
        up_in_c_list=[512, 768, 768, 672],
        nf_factor=1.5,
        norm_type="NormSpectral"),
    post_layers=dict(
        type='PostLayer',
        ni=300,
        last_cross=True,
        n_classes=3,
        bottle=False,
        norm_type="NormSpectral",
        y_range=(-3.0, 3.0)))

train_cfg = dict()
test_cfg = dict()

dataset_type = ''

train_pipeline = []
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt_img',
        # flag='grayscale',
        backend='pillow'
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=(35 * 16, 35 * 16),
        # scale=(25 * 16, 25 * 16),
        keep_ratio=False,
        backend='pillow'
    ),
    dict(
        type='RescaleToZeroOne',
        keys=['gt_img']
    ),
    # dict(
    #     type='ChannelsFromOneToThree',
    #     keys=['gt_img'],
    # ),
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


