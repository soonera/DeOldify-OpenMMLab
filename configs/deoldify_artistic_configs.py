custom_imports = dict(
    imports=['models', 'apis'],
    allow_failed_imports=False)

model = dict(
    type='DeOldify',
    generator=dict(
        type='DeOldifyGenerator',
        encoder=dict(
            type='ColorizationResNet',
            num_layers=34,
            pretrained=None,
            out_layers=[2, 5, 6, 7]),
        mid_layers=dict(
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
            y_range=(-3.0, 3.0))),
    discriminator=dict(
        type='DeOldifyDiscriminator',
        n_channels= 3,
        nf=256,
        n_blocks=3,
        p=0.15),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'29': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1e-2,
        style_weight=0,
        criterion='mse'),
    )

train_cfg = dict()
test_cfg = dict(show_input=True)

dataset_type = ''

train_pipeline = []
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img_gray',
        # flag='grayscale',
        backend='pillow'
    ),
    dict(
        type='Resize',
        keys=['img_gray'],
        scale=(35 * 16, 35 * 16),
        # scale=(25 * 16, 25 * 16),
        keep_ratio=False,
        backend='pillow'
    ),
    dict(
        type='RescaleToZeroOne',
        keys=['img_gray']
    ),
    dict(
        type='Normalize',
        keys=['img_gray'],
        mean=[0.4850, 0.4560, 0.4060],
        std=[0.2290, 0.2240, 0.2250],
        to_rgb=False),
    dict(
        type='Collect',
        keys=['img_gray'],
        meta_keys=['img_gray_path']),
    dict(type='ImageToTensor',
         keys=['img_gray']),
]

demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img_gray',
        # flag='grayscale',
        backend='pillow'
    ),
    dict(
        type='Resize',
        keys=['img_gray'],
        scale=(35 * 16, 35 * 16),
        # scale=(25 * 16, 25 * 16),
        keep_ratio=False,
        backend='pillow'
    ),
    dict(
        type='RescaleToZeroOne',
        keys=['img_gray']
    ),
    dict(
        type='Normalize',
        keys=['img_gray'],
        mean=[0.4850, 0.4560, 0.4060],
        std=[0.2290, 0.2240, 0.2250],
        to_rgb=False),
    dict(
        type='Collect',
        keys=['img_gray'],
        meta_keys=['img_gray_path']),
    dict(type='ImageToTensor',
         keys=['img_gray']),
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
