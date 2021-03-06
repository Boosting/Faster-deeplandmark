# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import PIL
import datasets.mydata
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class mydata(datasets.imdb):
    def __init__(self, image_set,annotationpath,devkit_path=None):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._annotationpath=annotationpath
        self._year = '2015'#year #2016.4.23
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', 'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        
        self._image_index = self._load_image_set_index(self._annotationpath)
        #print self._image_index
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def get_image_indix(self,all_index):
        image_index=[]
        numline=0
        while numline<len(all_index):
            line=all_index[numline]
            if int(line[1])==0:
               image_index.append(line[0])
               numline=numline+1
            else:
               image_index.append(line[0])
               numline=numline+int(line[1])
        return image_index
          


    def _load_image_set_index(self,imagelist):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path,imagelist)
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            full_index =[x.strip().split() for x in f.readlines()]
            #[line[0] for line in full_index]
            image_index=self.get_image_indix(full_index)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_annotation()
        #print gt_roidb
        #input("a:")

        #with open(cache_file, 'wb') as fid:
           # cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if  self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
            
        else:
            roidb = self._load_selective_search_roidb(None)
        #with open(cache_file, 'wb') as fid:
           # cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb): #faster rcnn dose not use mat at all
        filename = os.path.abspath(os.path.join(self._data_path,self.name + '.mat'))
        print filename
        
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['bboxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
       
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_annotation(self):
        """
        Load image and bounding boxes info from txt file        
        """
        annotationfile = os.path.join(self._data_path,self._annotationpath)
        with open(annotationfile) as f:
           split_line=[line.strip().split() for line in f.readlines() ]
        #print len(split_line)
        gt_roidb=[]
        numline=0
        
        while numline<len(split_line):
           line=split_line[numline]

           num_objs = int(line[1])
           if num_objs==0:
               boxes = np.zeros((num_objs+1, 4), dtype=np.uint16)
               gt_classes = np.zeros((num_objs+1), dtype=np.int32)
               overlaps = np.zeros((num_objs+1, self.num_classes), dtype=np.float32)#
           else:
               boxes = np.zeros((num_objs, 4), dtype=np.uint16)
               gt_classes = np.zeros((num_objs), dtype=np.int32)
               overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)#
           
           

           if num_objs==0:

               imgpath_neg = os.path.join(self._devkit_path,split_line[numline][0])
	       print imgpath_neg #
               w_neg = PIL.Image.open(imgpath_neg).size[0]
               h_neg = PIL.Image.open(imgpath_neg).size[1]
               boxes[0,:]=[0,0,w_neg-1,h_neg-1]
               gt_classes[0] = 0
               overlaps[0, 0] = 1.0
               numline=numline+1
              
           else:
               for anno_ix in range(0,num_objs):
                   l=split_line[numline+anno_ix]
                   cls=l[2]

                   #x1 = float(l[4])#######
                   #y1 = float(l[3])#######
                   #x2 = float(l[6])#######
                   #y2 = float(l[5])#######
		   #because train hk person 18481,so  we change the code @2016.8.16
                   x1 = float(l[3])#######
                   y1 = float(l[4])#######
                   x2 = float(l[5])#######
                   y2 = float(l[6])#######
                   if x1 > x2:
                      x1,x2 = x2,x1
                      print(1)
                   if y1 > y2:
                      y1,y2 = y2,y1
                      print(2)
                   imgpath = os.path.join(self._devkit_path,l[0])
		   print imgpath #
                   w = PIL.Image.open(imgpath).size[0]
                   h = PIL.Image.open(imgpath).size[1]
                   if x2 > w:
                      print 'x2>w : ',x2,w
                      print(imgpath)
                      x2 = w
                   if y2 > h:
                      print 'y2>h : ',y2,h
                      print(imgpath)
                      y2 = h
                   box_ix = anno_ix
                   boxes[box_ix, :] = [x1, y1, x2, y2]
                   gt_classes[box_ix] = cls
                   overlaps[box_ix, cls] = 1.0
               numline=numline+num_objs

           overlaps = scipy.sparse.csr_matrix(overlaps)
           gt_roidb.append({'boxes': boxes,
                          'gt_classes': gt_classes,
                          'gt_overlaps': overlaps,
                          'flipped': False})
                   
        return  gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename) 
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.mydata('trainval', '/home/vision/xc/py-faster-rcnn/data/xcMYdevkit')
    res = d.roidb
    from IPython import embed; embed()
