# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

from apis.colorization_inference import colorization_inference, init_colorization_model
from glob import glob
from os.path import join
from utils.video_process import VideoColorizer


def parse_args():
    parser = argparse.ArgumentParser(description='colorization demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='path to input video file')
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

    video_colorizer = VideoColorizer(model, args.work_dir)
    render_factor = 21

    video_paths = sorted(glob(join(args.work_dir, "source/*.mp4")))
    for video_path in video_paths:
        video_name = video_path.split('/')[-1]
        result_path = video_colorizer.colorize_from_file_name(video_name, render_factor=render_factor)
        print(result_path)


if __name__ == '__main__':
    main()