# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class ChannelsFromOneToThree:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = np.repeat(results[key], 3, axis=2)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
