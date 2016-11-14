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
from nmsly import nmsly


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg1024': ('VGG_CNN_M_1024', 'VGG_CNN_M_1024_faster_rcnn_final.caffemodel')}

def demo(line, image_name, imgSavePath, detnet):#,clsnet,transformer,labels):
    """Detect object classes in an image using pre-computed object proposals."""
    im = cv2.imread(image_name)
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(detnet, im)
    #print ('Detection took {:.3f}s for '
           #'{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.6

    cls_boxes = boxes[:, 4:8]
    cls_scores = scores[:, 1]
    dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nmsly(dets, NMS_THRESH)
    dets = dets[keep, :]

    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

    #print "number of boxes is:",len(inds)
 
    #im = im[:, :, (2, 1, 0)]

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),2,8,0)
        text="%f "%score
        cv2.putText(im,text,(int(bbox[0]),int(bbox[1]-10)),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
        save_name = line.split('/')[1] 
        cv2.imshow("face" ,im)
        cv2.imwrite(os.path.join(imgSavePath, str(save_name)),im)
    """
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

               
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show()
   
    save_name = line.split('/')[1]    
    plt.savefig(os.path.join(imgSavePath, str(save_name)))        
    """   


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf') #############

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    model_file = '/home/vision/xc/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'   

    model_def_file = '/home/vision/xc/py-faster-rcnn/output/default/trainval/ZF_faster_rcnn_final.caffemodel'

    imgPath = '/home/vision/xc/py-faster-rcnn/data/test/Face_test/lfpw_test_249_bbox.txt'

    imgSavePath = '/home/vision/xc/py-faster-rcnn/output/Face_out_images/'
   

    if not osp.isfile(model_file):
        raise IOError(('{:s} not found').format(model_file))
    if not os.path.isfile(model_def_file):
        raise IOError(('{:s} not found\n').format(model_def_file))
    if not osp.exists(imgPath):
        print("imgPath path not exist")
    if (os.path.isdir(imgSavePath)==False):
         os.makedirs(imgSavePath)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    detnet = caffe.Net(model_file, model_def_file, caffe.TEST)  
    print '\n\nLoaded network {:s}'.format(model_def_file)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(detnet, im)


    with open(imgPath) as f:
           split_line=[line.strip() for line in f.readlines() ]

   
    num=1
    for line1 in split_line:      
      strlen = line1.split()
      line = strlen[0]     
      image_name='/home/vision/xc/py-faster-rcnn/data/test/Face_test/' + line 
      demo(line, image_name, imgSavePath, detnet)
      print "now save with No.", num, line
      num=num+1
     

