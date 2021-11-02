# DeOldifyPredictor


## Introduction
English | [简体中文](/README_zh-CN.md)

This repo provides DeOldify model for image/video colorization using [mmcv](https://github.com/open-mmlab/mmcv) and [mmediting](https://github.com/open-mmlab/mmediting). About more details of DeOldify, please refer to https://github.com/jantic/DeOldify

### Major features of this repo
- **Modular design**: We decompose the deoldify framework into different components and one can easily construct a customized editor framework by combining different modules.

![](https://i.imgur.com/58BVejq.png)

## Examples

Orignal Image              |  Stable Colorization Image         |  Artistic Colorization Image
:-------------------------:|:-------------------------:         |:-------------------------:
![](https://i.imgur.com/lpiGyel.jpg)  |  ![](https://i.imgur.com/Y1pqmTT.png) | ![Artistic](https://i.imgur.com/TaBEP3B.png)


## Installation

Please refer to https://github.com/open-mmlab/mmediting/blob/master/docs/install.md for installation.


## License

All code in this repository is under the [MIT license](LICENSE).



## Get Started

### Completed Generator Weights
These weights are from https://github.com/jantic/DeOldify
- [Artistic](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth)
- [Stable](https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=0)
- [Video](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth)

The weight keys will be automatically transformed  in this [file](./apis/colorization_inference.py). You should only put these three weight files in ./checkpoints



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

- If you want to use stable mode:
```shell
python demo/image_demo.py \
    work_dirs/stable/source/1.jpg \
    configs/deoldify_stable_configs.py \
    checkpoints/ColorizeStable_gen.pth \
    --out work_dirs/stable/result/1.png \
    --show
```
The predicted stable colorization result will be save in `work_dirs/stable/result/1.png`.

- If you want to use artistic mode:
```shell
python demo/image_demo.py \
    work_dirs/artistic/source/1.jpg \
    configs/deoldify_artistic_configs.py \
    checkpoints/ColorizeArtistic_gen.pth \
    --out work_dirs/artistic/result/1.png \
    --show
```
The predicted artistic colorization result will be save in `work_dirs/artistic/result/1.png`.


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
- You can only use video mode:
```shell
python demo/video_demo.py \
    work_dirs/video/source/test.mp4 \
    configs/deoldify_video_configs.py \
    checkpoints/ColorizeVideo_gen.pth \
    --out work_dirs/video/result/test.mp4 \
    --show
```

The predicted video colorization result will be saved in `work_dirs/video/result/test.mp4`.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{DeOldify-OpenMMLab,
    title={DeOldify Implement using OpenMMLab},
    author={DeOldify-OpenMMLab Contributors},
    howpublished = {\url{https://github.com/soonera/DeOldify-OpenMMLab}},
    year={2021}
}
```


