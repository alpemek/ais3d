from __future__ import print_function
from show3d_balls import *
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
from My_datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import itertools
#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '/home/emeka/Schreibtisch/AIS/ais3d/Pointnet/seg/seg_model_24.pth',  help='model path')
parser.add_argument('--idx', type=int, default = 0,   help='model index')



opt = parser.parse_args()
print (opt)

d = PartDataset(root="/home/dllab/kitti_object/data_object_image_2",image_sets=[('train')],train = False)
GT_score_arr_ped = []
GT_score_arr_cyc = []
GT_score_arr_car = []

pred_score_arr_ped = []
pred_score_arr_cyc = []
pred_score_arr_car = []
#idx = opt.id
#PEDESTRIAN, TRY 150 OR 166, THEZ CONTAIN PEDESTRIAN 10
print(len(d))
for idx in range(100,len(d)):
    print("model %d/%d" %( idx, len(d)))
    print(d.ids[idx])
    point, seg, point_nn = d[idx]
    print(point.size(), seg.size())

    point_np = point.numpy()
    seg_num=seg.numpy()
    seg_num = seg_num.tolist()
    seg_num=list(itertools.chain.from_iterable(seg_num))
    seg_num=np.asarray(seg_num)
   # print(seg.numpy())

    #THESE ARE THE COLORS AVAILABLE https://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.cm.get_cmap("Spectral", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:,:3]
    gt = cmap[seg.numpy() - 1, :]
    gt = np.squeeze(gt)
    classifier = PointNetDenseCls(k = 4)
    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    point = point.transpose(1,0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    #print('Point')
    #print( point)
    #print('---------------')

    pred, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]
    np.set_printoptions(threshold=np.nan)

    #print(pred.data.max)
    #print(pred_choice.numpy()[0])
    #print('################')
    #print(seg.numpy())
    #print('################')
    #print( (pred.data).numpy())
    #print('################')

    pred_choice_num = pred_choice.numpy()[0]
    seg_leveled = seg_num - 1

    #seg_leveled = np.array(seg_leveled.tolist())

    totalnum_points = seg_leveled.size
    obj_index = np.nonzero(seg_leveled)

    print('##################')

    obj_points = seg_leveled[obj_index]
    obj_num = obj_points.size
    GT_score= obj_num/seg_num.size
    pred_point = pred_choice_num[obj_index]
    if obj_num == 0:
        continue
   # GT_score = [0,0,0]
   # GT_score[1] = 0.5*GT_score[1]+0.5
    print('GT Score: {}'.format(GT_score))
    pred_err = pred_point - obj_points
    match_num = np.nonzero(pred_err == 0)[0]
    pred_score = (match_num).size /obj_num
    indd_cls = (seg_leveled[obj_index]) [0]
    if indd_cls == 1:
        pred_score_arr_cyc.append(pred_score)
        GT_score_arr_cyc.append(GT_score)

        print('Pred Score Cyc {}'.format(np.mean(np.asarray(pred_score_arr_cyc))))
        print('GT Score Cyc {}'.format(np.mean(np.asarray(GT_score_arr_cyc))))
    elif indd_cls == 2:
        pred_score_arr_car.append(pred_score)
        GT_score_arr_car.append(GT_score)
        print('Pred Score Car {}'.format(np.mean(np.asarray(pred_score_arr_car)) ))
        print('GT Score Car {}'.format(np.mean(np.asarray(GT_score_arr_car)) ))
    elif indd_cls == 3:
        pred_score_arr_ped.append(pred_score)
        GT_score_arr_ped.append(GT_score)
        print('Pred Score Ped {}'.format(np.mean(np.asarray(pred_score_arr_ped))) )
        print('GT Score Ped {}'.format(np.mean(np.asarray(GT_score_arr_ped)) ))

    print('Prediction Score: {:.4f}'.format(pred_score))
    pred4=(pred.data).numpy()

    #print(pred_choice.size())
    pred_color = cmap[pred_choice.numpy()[0], :]

    print(point_np.shape)
    print(gt.shape)
    print(pred_color.shape)

    #showpoints(point_nn, gt, pred_color,ballradius=1)
print('END')
print('Pred Score Cyc {}'.format(np.mean(np.asarray(pred_score_arr_cyc))))
print('GT Score Cyc {}'.format(np.mean(np.asarray(GT_score_arr_cyc))))
print('Pred Score Car {}'.format(np.mean(np.asarray(pred_score_arr_car)) ))
print('GT Score Car {}'.format(np.mean(np.asarray(GT_score_arr_car)) ))
print('Pred Score Ped {}'.format(np.mean(np.asarray(pred_score_arr_ped))) )
print('GT Score Ped {}'.format(np.mean(np.asarray(GT_score_arr_ped)) ))