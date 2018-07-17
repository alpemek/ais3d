from __future__ import print_function
#from show3d_balls import *
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import itertools

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '/home/emeka/Schreibtisch/AIS/ais3d/Pointnet_Box/seg/seg_model_99.pth',  help='model path')
parser.add_argument('--idx', type=int, default = 0,   help='model index')
opt = parser.parse_args()
print (opt)

d = PartDataset(root="/home/dllab/kitti_object/data_object_image_2",image_sets=[('val')],train = False)


#PEDESTRIAN, TRY 150 OR 166, THEZ CONTAIN PEDESTRIAN 10
print(len(d))
for idx in range(1,len(d)):
    print("model %d/%d" %( idx, len(d)))
    print(d.ids[idx])
    point, seg = d[idx]
    print(point.size(), seg.size())

    point_np = point.numpy()

    seg_num=seg.numpy()


    classifier = PointNetDenseCls(k = 7)
    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    point = point.transpose(1,0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    #print('Point')
    #print( point)
    #print('---------------')

    pred, _ = classifier(point)
    
    pred = pred.view(-1, 7)
    pred_num = pred.data.numpy()
    print('Pred {}'.format(pred_num[0,:]))
    print('Target {}'.format(seg_num[0,:]))
    err = np.abs(pred_num - seg_num) /np.abs(seg_num)
    err_col = np.average(err,0)
    print('Error:')
    print(err_col)
    #err_per = err_col/ np.abs( np.average(seg_num,0))
    #print(err_per)
    print('#########')
