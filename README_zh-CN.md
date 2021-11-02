# DeOldifyPredictor


## 介绍
[English](/README.md) | 简体中文

本项目基于[mmcv](https://github.com/open-mmlab/mmcv) and [mmediting](https://github.com/open-mmlab/mmediting) 实现了DeOldiy的推理部分，可用于图片和视频的上色。
更多关于DeOldify的介绍，可参考原作者的github: https://github.com/jantic/DeOldify

### 本实现的主要特点
- **模块化设计**： 我们将deoldify框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

## 效果展示

原黑白图              |  稳定模式上色图         |  艺术模式上色图
:-------------------------:|:-------------------------:         |:-------------------------:
![](https://i.imgur.com/lpiGyel.jpg)  |  ![](https://i.imgur.com/Y1pqmTT.png) | ![Artistic](https://i.imgur.com/TaBEP3B.png)


## 安装

相比于原作者使用fastai， 本项目使用了OpenMMLab中的MMEditing，所需安装包与MMEditing一致，可参考 https://github.com/open-mmlab/mmediting/blob/master/docs/install.md


## 许可证
本项目开源自[MIT license](LICENSE)

## 开始使用

### 训练完成的模型
这些权重来自原作者https://github.com/jantic/DeOldify
- [Artistic](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth)
- [Stable](https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=0)
- [Video](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth)

权重的键名会在[本代码](./apis/colorization_inference.py) 中自动被转换。您只需要把这三个下载好的权重文件放在./checkpoints目录下

### 图片上色
对单张图片进行上色，命令行模板如下：
```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--out ${OUT}] \
    [--show ${SHOW}]
```

例子:

- 如果使用stable模式:
```shell
python demo/image_demo.py \
    work_dirs/stable/source/1.jpg \
    configs/deoldify_stable_configs.py \
    checkpoints/ColorizeStable_gen.pth \
    --out work_dirs/stable/result/1.png \
    --show
```
预测出的stable模式上色结果将会保存为`work_dirs/stable/result/1.png`.

- 如果使用artistic模式:
```shell
python demo/image_demo.py \
    work_dirs/artistic/source/1.jpg \
    configs/deoldify_artistic_configs.py \
    checkpoints/ColorizeArtistic_gen.pth \
    --out work_dirs/artistic/result/1.png \
    --show
```
预测出的artistic模式上色结果将会保存为`work_dirs/artistic/result/1.png`.


### 视频上色
对单个视频进行上色，命令行模板如下：
```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--out ${OUT}] \
    [--show ${SHOW}]
```

例子:
- 只能使用video模式:
```shell
python demo/video_demo.py \
    work_dirs/video/source/test.mp4 \
    configs/deoldify_video_configs.py \
    checkpoints/ColorizeVideo_gen.pth \
    --out work_dirs/video/result/test.mp4 \
    --show
```
预测出的video模式上色结果将会保存为`work_dirs/video/result/test.mp4`.

## 引用

如果您觉得 DeOldiy-OpenMMLab 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{DeOldify-OpenMMLab,
    title={DeOldify Implement using OpenMMLab},
    author={DeOldify-OpenMMLab Contributors},
    howpublished = {\url{https://github.com/soonera/DeOldify-OpenMMLab}},
    year={2021}
}
```


