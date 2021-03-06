#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict data from http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
"""

import os, sys
from functools import partial
import cv2
from common import getDataFromTxt, createDir, logger, drawLandmark
from common import level1, level2, level3
######################################################
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe
import argparse
import os.path as osp
#from nmsly import nmsly


class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

if __name__ == '__main__':

    assert(len(sys.argv) == 2)
    level = int(sys.argv[1])
    if level == 0:
        P = partial(level1, FOnly=True)
    elif level == 1:
        P = level1
    elif level == 2:
        P = level2
    else:
        P = level3

##################################
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    model_file = '/home/vision/xc/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'   

    model_def_file = '/home/vision/xc/py-faster-rcnn/output/default/trainval/ZF_faster_rcnn_final.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)

    detnet = caffe.Net(model_file, model_def_file, caffe.TEST)  
    print '\n\nLoaded network {:s}'.format(model_def_file)
##################################
    #open camera to detect face,for real time
    capture=cv2.VideoCapture(0)
    print "VideoCapture is open!! :",capture.isOpened()

    while True:
        ret,img = capture.read()  
        scores, boxes = im_detect(detnet, img)       
        CONF_THRESH = 0.7
        NMS_THRESH = 0.6
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, 1]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if inds is None:
            continue
        for i in inds:
            bbox = dets[i, :4]#(0,2,1,3)
            cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),2,8,0)
            bbox[1],bbox[2] = bbox[2],bbox[1]       
            bbox = BBox(bbox)
        ##################
        cv2.imshow("real-time face detection" ,img) 

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmark = P(imgGray, bbox)
        landmark = bbox.reprojectLandmark(landmark)
        drawLandmark(img, bbox, landmark)
        cv2.imshow("real-time face-landmark detection", img)

        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    cv2.destroyAllWindows() ##############
