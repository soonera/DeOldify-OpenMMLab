# DeOldify-mmcv

This repo provides DeOldify model for image colorization using mmcv.

### Image demo
This script performs inference on a single image.
```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--out ${OUT}] \
    [--show ${SHOW}]
```

Examples:

```shell
python demo/image_demo.py work_dirs/example_exp_stable/source/1.jpg configs/deoldify_stable_configs.py checkpoints/ColorizeStable_gen.pth --out work_dirs/example_exp_video/result/1.png --show
```

The predicted inpainting result will be save in `work_dirs/example_exp_video/result/1.png`.


### Video demo
This script performs inference on a single video.
```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--out ${OUT}] \
    [--show ${SHOW}]
```

Examples:

```shell
python demo/video_demo.py work_dirs/example_exp_video/source/test.mp4 configs/deoldify_video_configs.py checkpoints/ColorizeVideo_gen.pth --out work_dirs/example_exp_video/result/test.mp4 --show
```

The predicted inpainting result will be save in `work_dirs/example_exp_video/result/test.mp4`.




