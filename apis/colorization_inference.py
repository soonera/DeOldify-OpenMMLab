# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import PIL
from PIL import Image

import mmcv
from mmcv.parallel import collate, scatter
from mmedit.models import build_model
from mmedit.datasets.pipelines import Compose


def init_colorization_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.model.pretrained = None
    # config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        # checkpoint = load_checkpoint(model, checkpoint)
        params = torch.load(checkpoint, map_location='cpu')

        # 自己加的
        keys_0 = model.state_dict().keys()
        # torch.save({'model': model.state_dict()}, 'keys.pth')

        keys_1 = params['model'].keys()
        print(keys_0 == keys_1)

        if keys_0 != keys_1 and len(keys_0) == len(keys_1):
            d = params['model'].items()
            d1 = model.state_dict().items()
            from collections import OrderedDict
            new_d = OrderedDict()
            for (k, v), (k1, v1) in zip(d, d1):
                new_d[k1] = v
            params['model'] = new_d

        model.load_state_dict(params['model'])

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def denorm(x, mean, std):
    x = x * std[..., None, None] + mean[..., None, None]
    return x


def overlay_color(raw_color, orig):
    color_y, color_u, color_v = raw_color.convert("YCbCr").split()
    orig_y, orig_u, orig_v = orig.convert("YCbCr").split()
    final = Image.merge("YCbCr", (orig_y, color_u, color_v)).convert("RGB")
    return final


def post_process(results, orig_img):
    # denom
    mean = torch.tensor([0.4850, 0.4560, 0.4060]).cuda()  # imagenet的均值和方差
    std = torch.tensor([0.2290, 0.2240, 0.2250]).cuda()
    results = denorm(results.detach(), mean, std)

    # clamp
    results = results.float().clamp(min=0, max=1)

    # To PIL
    out = (results.cpu().numpy()*255).astype('uint8').transpose(1, 2, 0)
    out = Image.fromarray(out)

    # Resize
    orig_image = None
    if isinstance(orig_img, str):
        orig_image = PIL.Image.open(orig_img).convert('RGB')
    if isinstance(orig_img, np.ndarray):
        orig_image = Image.fromarray(np.uint8(orig_img))

    # orig_image = PIL.Image.open(orig_img_path).convert('RGB')
    raw_color = out.resize(orig_image.size, resample=PIL.Image.BILINEAR)

    # overlay color
    final = overlay_color(raw_color, orig_image)

    # return final
    if isinstance(orig_img, str):
        return final
    if isinstance(orig_img, np.ndarray):
        return np.asarray(final)[:, :, ::-1]
    return None


def colorization_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        np.ndarray: The predicted colorization result.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    test_pipeline = Compose(cfg.test_pipeline)

    data = None
    if isinstance(img, str):
        data = dict(gt_img_path=img)
    if isinstance(img, np.ndarray):
        data = dict(gt_img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    model.eval()
    with torch.no_grad():
        results = model.forward(data['gt_img']).squeeze()

    final = post_process(results, img)

    return final
