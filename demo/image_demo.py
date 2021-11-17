# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import mmcv
from apis import (colorization_inference, init_colorization_model)
from mmedit.core import tensor2img


def parse_args():
    parser = ArgumentParser(description='Colorization image demo')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--out', type=str, help='Output image file')
    parser.add_argument('--show', action='store_true', help='Show image')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_colorization_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    output = colorization_inference(model, args.img)
    output = tensor2img(output)
    
    # show the results
    if args.show:
        mmcv.imshow(output, 'predicted colorization result')

    # save
    if isinstance(args.out, str):
        mmcv.imwrite(output, args.out)


if __name__ == '__main__':
    args = parse_args()
    main(args)
