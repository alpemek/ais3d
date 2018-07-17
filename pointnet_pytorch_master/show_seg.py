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
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))
np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '/home/emeka/Schreibtisch/AIS/ais3d/pointnet.pytorch-master/seg/seg_model_0.pth',  help='model path')
parser.add_argument('--idx', type=int, default = 0,   help='model index')



opt = parser.parse_args()
print (opt)

d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair','Car','Airplane'], train = False)

#idx = opt.id
idx =280;
print("model %d/%d" %( idx, len(d)))

point, seg = d[idx]
print(point.size(), seg.size())

point_np = point.numpy()


#THESE ARE THE COLORS AVAILABLE https://matplotlib.org/examples/color/colormaps_reference.html
cmap = plt.cm.get_cmap("Spectral", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg.numpy() - 1, :]

classifier = PointNetDenseCls(k = 4)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

point = point.transpose(1,0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
print('Point')
print( point)
print('---------------')
pred, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice.numpy())
print('################')
print( (pred.data).numpy())



#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

print(point_np.shape)
print(gt.shape)
print(pred_color.shape)

showpoints(point_np, gt, pred_color)

