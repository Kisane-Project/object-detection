
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from random import shuffle

@PIPELINES.register_module()
class LoadImage:
    def __call__(self, file_name=''):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        results = {}
        results['filename'] = file_name
        results['ori_filename'] = file_name
        img = mmcv.imread(file_name)
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results