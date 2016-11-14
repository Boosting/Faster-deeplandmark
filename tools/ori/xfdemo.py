#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os.path as osp
#from lzhextraction import init_net
#from lzhextraction import forward

CLASSES = ('__background__',
           'target')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

ALLOWED_IMAGE_EXTENSIONS = set(['.png', '.bmp', '.jpg', '.jpe', '.jpeg', '.gif', '.tif'])

def is_allowed_image(file):
    _,ext = osp.splitext(file)
    return ext in ALLOWED_IMAGE_EXTENSIONS

def vis_detections(fileout, num, outpath, image_name, im, class_name, dets, thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    #print ('det-Result-Show:')
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)

        text="%f "%score
        fileout.write("%d,%d,%d,%d,%d,%f\n" %(num,bbox[0],bbox[1],bbox[2]-bbox[0]+1,bbox[3]-bbox[1]+1,score))
                
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show()
    save_name = image_name[:-4]+'_'+class_name+'.jpg'
    print "now deal with:",image_name[:-4]
    plt.savefig(os.path.join(outpath,str(save_name)))



#region_class() is for xiaofei to classify and save the region of a image, but for us do pedestrian, we do not use this function~~~~
def region_class(clsnet,transformer,labels, outpath, image_name, im, class_name, dets, thresh):

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print ('no detRes region ')
        predict_label = 'aaa_failed'
        dst_dir = osp.join(outpath, predict_label)
        if not osp.exists(dst_dir):
            os.makedirs(dst_dir)
        cv2.imwrite((dst_dir + image_name),im)
        return
    im = im[:, :, (2, 1, 0)]
    max_score = 0
    # find the max scores~
    for i in inds:
        bbox = dets[i, :4][0:4]
        region = im[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        #cv2.imshow("region",region)
        scores = forward(clsnet, region, transformer)
        max_score_1 = np.amax(scores)
        if max_score_1 > max_score:
            max_score = max_score_1

    if max_score > .65:
        predict_label = labels[np.argmax(scores)]
    else:
        predict_label = 'aaa_unknown'

    dst_dir = osp.join(outpath, predict_label)
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    cv2.imwrite((dst_dir +'/'+ image_name),region)

def demo(fileout, num, inpath, outpath, detnet, image_name):#,clsnet,transformer,labels):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the test image
    image_filenames = [inpath + image_name]
    im_file = image_filenames[0]
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(detnet, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
           #'{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.88
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #region_class(clsnet,transformer,labels, outpath, image_name, im, cls, dets, thresh=CONF_THRESH)
        vis_detections(fileout, num, outpath, image_name, im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

def main():

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    model_file = os.path.join('/home/vision/xc/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt')
    
    model_def_file = os.path.join('/home/vision/xc/py-faster-rcnn/output/default/trainval/ZF_faster_rcnn_final.caffemodel')
    inpath = os.path.join('/home/vision/xc/py-faster-rcnn/data/INRIA_test/')
    outpath = os.path.join('/home/vision/xc/py-faster-rcnn/output/out_images/')
    fileoutpath = "/home/vision/xc/py-faster-rcnn/output/INRIA.txt"

    if not osp.isfile(model_file):
        raise IOError(('{:s} not found').format(model_file))
    if not os.path.isfile(model_def_file):
        raise IOError(('{:s} not found\n').format(model_def_file))
    if not osp.exists(inpath):
        print("input path not exist")
        return 1
    if not osp.isdir(inpath):
        print("input path not a directory")
        return 1
    if (os.path.isdir(outpath)==False):
         os.makedirs(outpath)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    fileout= open(fileoutpath,"w")
    detnet = caffe.Net(model_file, model_def_file, caffe.TEST)
    #clsnet,transformer,labels = init_net()
    print '\n\nLoaded network {:s}'.format(model_def_file)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(detnet, im)

    for root, _, files in os.walk(inpath):
        print 'processing', root
        num = 1
        for f in files:
            source_file = osp.join(root, f)
            if not is_allowed_image(source_file):
                continue

            dirpath, dst_name = osp.split(source_file)
            _, dir_name = osp.split(dirpath)
            demo(fileout, num, inpath, outpath, detnet, f)#,clsnet,transformer,labels)
            num += 1
    fileout.close()

if __name__ == '__main__':
    main()
