# coding: utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""



import datasets.pascal_voc
import datasets.mydata
import numpy as np

__sets = {}
imageset="trainval"#图片集性质 train，test，trainval
devkit="/home/vision/xc/py-faster-rcnn/data/xcMYdevkit"#标注文档，图片根目录
annotationpath="Annotations/annotation.txt"#标注文件所在目录



def get_imdb(name):
    """Get an imdb (image database) by name."""
    
    __sets["mydata"]=(lambda imageset = imageset, devkit = devkit: datasets.mydata(imageset,annotationpath,devkit))
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()


