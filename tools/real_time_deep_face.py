import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os.path as osp
from nmsly import nmsly

  
if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    model_file = '/home/vision/xc/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'   

    model_def_file = '/home/vision/xc/py-faster-rcnn/output/default/trainval/ZF_faster_rcnn_final.caffemodel'

    caffe.set_mode_gpu()
    caffe.set_device(0)

    detnet = caffe.Net(model_file, model_def_file, caffe.TEST)  
    print '\n\nLoaded network {:s}'.format(model_def_file)

    #open camera to detect face,for real time
    capture=cv2.VideoCapture(0)
    print capture.isOpened()

    while True:
        ret,im = capture.read()                   
        #im = cv2.imread(image_name)    
        scores, boxes = im_detect(detnet, im)

        CONF_THRESH = 0.7
        NMS_THRESH = 0.6

        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, 1]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nmsly(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),2,8,0)
 
        cv2.imshow("real-time face detection" ,im) 
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    cv2.destroyAllWindows()     

