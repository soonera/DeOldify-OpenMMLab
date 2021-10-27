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
    parser.add_argument('work_dir', help='path to input video file')
    # parser.add_argument('save_path', help='path to save colorization result')
    # parser.add_argument(
    #     '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

# ffmpeg 方法
# from utils.video_process_ffmpeg import VideoColorizer
# def main():
#     args = parse_args()
#
#     model = init_colorization_model(
#         args.config, args.checkpoint, device=torch.device('cuda', args.device))
#
#     video_colorizer = VideoColorizer(model, args.work_dir)
#     render_factor = 21
#
#     video_paths = sorted(glob(join(args.work_dir, "source/*.mp4")))
#     for video_path in video_paths:
#         video_name = video_path.split('/')[-1]
#         result_path = video_colorizer.colorize_from_file_name(video_name, render_factor=render_factor)
#         print(result_path)

# opencv 方法
import cv2
import numpy as np
import os


def main():
    args = parse_args()

    model = init_colorization_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # video_colorizer = VideoColorizer(model, args.work_dir)
    # render_factor = 21

    video_paths = sorted(glob(join(args.work_dir, "source/*.mp4")))
    for video_path in video_paths:
        video_name = video_path.split('/')[-1]
        save_path = join(args.work_dir, "result", video_name)
        capture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = capture.get(5)
        size = (int(capture.get(3)), int(capture.get(4)))  # 宽度、高度
        writer = cv2.VideoWriter(save_path, fourcc, fps, size, True)
        i = 0
        while True:
            ret, img_src = capture.read()
            if not ret:
                break
            temp_path = join(args.work_dir, "temp/1.png")
            cv2.imwrite(temp_path, img_src)
            img_out = colorization_inference(model, temp_path)
            writer.write(np.asarray(img_out)[:,:,::-1])
            os.remove(temp_path)
            i += 1
            print(i)
        writer.release()
        print('ok')


if __name__ == '__main__':
    main()