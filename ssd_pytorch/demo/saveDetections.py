import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
from ssd import build_ssd
from data import KITTI_CLASSES as labels
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform, KITTIDetection, KITTI_ROOT, KITTIAnnotationTransform

net = build_ssd('test', 300, 4)    # initialize SSD
net.load_weights('../weights/ssd300_Resz_KITTI_80000.pth')

testset = KITTIDetection(KITTI_ROOT,['train'], None, KITTIAnnotationTransform())
save_folder = '/home/emeka/Schreibtisch/AIS/ais3d/Detections/'
for img_id in range(0,len(testset)):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()

    x = torch.from_numpy(x).permute(2, 0, 1)
    t = time.time()
    x_var = Variable(x.unsqueeze(0))     # wrap tensor in Variable

    if torch.cuda.is_available():
        x_var = x_var.cuda()
    y = net(x_var)
    elapsed = time.time() - t
   
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    filename = save_folder+'{}.txt'.format(testset.img_id_return(img_id))
    f = open(filename,'w')
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] > 0.3:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            print(coords)
            f.write('{} {} {} {} {}'.format(label_name,pt[0],pt[1],pt[2],pt[3])  + '\n')

            j+=1
    f.close()
