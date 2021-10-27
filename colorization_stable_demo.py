# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

from apis.colorization_inference import colorization_inference, init_colorization_model
from glob import glob
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(description='colorization demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='path to input image file')
    # parser.add_argument('save_path', help='path to save colorization result')
    # parser.add_argument(
    #     '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_colorization_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_paths = sorted(glob(join(args.work_dir, "source/*.*")))
    for img_path in img_paths:

        output = colorization_inference(model, img_path)

        file_name = img_path.split('/')[-1].split('.')[0]
        output.save(join(args.work_dir, "result", file_name + '.png'))
        print(file_name)


if __name__ == '__main__':
    main()
