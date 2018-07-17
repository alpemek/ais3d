from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path as osp
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json

KITTI_ROOT = osp.join("/home/dllab/kitti_object/data_object_image_2")

class PartDataset(data.Dataset):
    def __init__(self, root=osp.join("/home/dllab/kitti_object/data_object_image_2"),
                 npoints = 2500,image_sets=[('train')], classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.image_set = image_sets
        self.num_seg_classes = 4
        self._annopath = '/home/emeka/Schreibtisch/AIS/ais3d/PCD_Files/Labeled'

       # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
       # self.cat = {}

        self.classification = classification

        self.ids = list()
        for ( name) in image_sets:
            rootpath = osp.join(self.root)
            for line in open(osp.join(rootpath, name + '.txt')):
                ll=line.strip()
                label_filename = "{}/{}.txt".format(self._annopath,ll)
                if os.path.isfile(label_filename):
                    self.ids.append((ll))
                    #print(ll)

       # with open(self.catfile, 'r') as f:
       #     for line in f:
       #         ls = line.strip().split()
       #         self.cat[ls[0]] = ls[1]
        #print(self.cat)
       # if not class_choice is  None:
       #     self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

       # self.meta = {}
       # for item in self.cat:
            #print('category', item)
       #     self.meta[item] = []
       #     dir_point = os.path.join(self.root, self.cat[item], 'points')
       #     dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
       #     fns = sorted(os.listdir(dir_point))
       #     if train:
       #         fns = fns[:int(len(fns) * 0.9)]
       #     else:
       #         fns = fns[int(len(fns) * 0.9):]#

            #print(os.path.basename(fns))
       #     for fn in fns:
       #         token = (os.path.splitext(os.path.basename(fn))[0])
       #         self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

       # self.datapath = []
       # for item in self.cat:
       #     for fn in self.meta[item]:
       #         self.datapath.append((item, fn[0], fn[1]))


       # self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
       # print(self.classes)
       # self.num_seg_classes = 0
       # if not self.classification:
       #     for i in range(len(self.datapath)//50):
       #         l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
       #         if l > self.num_seg_classes:
       #             self.num_seg_classes = l
        #print(self.num_seg_classes)



    def __getitem__(self, index):
        #my Code
        img_id = self.ids[index]
        label_filename = "{}/{}.txt".format(self._annopath,img_id)
        #print(label_filename)
        complete_set = np.loadtxt(label_filename).astype(np.float32)

        point_set = complete_set[:,0:3]
        cls = complete_set[:,-1:]
        seg =np.int64( complete_set[:,-1:])
        #print('1. Points_set{} , seg={} '.format(point_set.shape,seg.shape))
        #Added New
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        point_set = point_set / np.absolute(point_set).max(axis=0)
        seg = seg[choice]
       # print('2. Points{} , seg={} '.format(point_set.shape,seg.shape))

       # print('seg'+seg.shape)
       # print(point_set.shape)
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
		
        #my Code Ends

#        seg = np.loadtxt(fn[2]).astype(np.int64)
#        #print(point_set.shape, seg.shape)#

#        choice = np.random.choice(len(seg), self.npoints, replace=True)
#        #resample
#        point_set = point_set[choice, :]
#        seg = seg[choice]

#        point_set = torch.from_numpy(point_set)
#        seg = torch.from_numpy(seg)
#        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

  #  def __getitem__(self, index):
  #      fn = self.datapath[index]
  #      cls = self.classes[self.datapath[index][0]]
  #      point_set = np.loadtxt(fn[1]).astype(np.float32)
  #      seg = np.loadtxt(fn[2]).astype(np.int64)
  #      #print(point_set.shape, seg.shape)

#        choice = np.random.choice(len(seg), self.npoints, replace=True)
#        #resample
#        point_set = point_set[choice, :]
#        seg = seg[choice]
#        point_set = torch.from_numpy(point_set)
#        seg = torch.from_numpy(seg)
#        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#        if self.classification:
#            return point_set, cls
#        else:
#            return point_set, seg

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
