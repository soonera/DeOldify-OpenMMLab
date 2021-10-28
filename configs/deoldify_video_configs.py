custom_imports = dict(
    imports=['models', 'apis'],
    allow_failed_imports=False)

model = dict(
    type='DeOldify',
    encoder=dict(
        type='ColorizationResNet',
        num_layers=101,
        pretrained=None,
        out_layers=[2, 5, 6, 7]),
    mid_layers=dict(
        # channel_factors=[2, 1],
        type='MidConvLayer',
        norm_type="NormSpectral",
        ni=2048),
    decoder=dict(
        type='UnetWideDecoder',
        self_attention=True,
        x_in_c_list=[64, 256, 512, 1024],
        ni=2048,
        nf_factor=2,
        norm_type="NormSpectral",),
    post_layers=dict(
        type='PostLayer',
        ni=256,
        last_cross=True,
        n_classes=3,
        bottle=False,
        norm_type="NormSpectral",
        y_range=(-3.0, 3.0)),
    )

train_cfg = dict()
test_cfg = dict()

dataset_type = ''

train_pipeline = []
test_pipeline = [
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=(160, 160),
        keep_ratio=False,
        # backend='pillow'
        backend='cv2'
    ),
    dict(
        type='RescaleToZeroOne',
        keys=['gt_img']
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
        meta_keys=['gt_img']),
    dict(
        type='ImageToTensor',
        keys=['gt_img']),
]


