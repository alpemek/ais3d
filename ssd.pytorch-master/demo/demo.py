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

net = build_ssd('test', 300, 4)    # initialize SSD
#net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
#net.load_weights('../weights/VOC.pth')
net.load_weights('../weights/ssd300_KITTI_110000.pth')
#FOR TESTING WITH ONLY ONE IMAGE, WITHOUT THE DATASET FOLDER, COMMENT THE LIGHT ABOVE AND UNCOMMENT LINE BELOW. ALSO COMMENT THE LINE BELOW image = testset.pull_image(img_id)
#image = cv2.imread('/home/emeka/Downloads/pp.jpeg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform, KITTIDetection, KITTI_ROOT, KITTIAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
#testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform()) #CHANGED
testset = KITTIDetection(KITTI_ROOT, None, KITTIAnnotationTransform())

for img_id in range(10,20):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    #plt.figure(figsize=(10,10))
    #plt.imshow(rgb_image)
    #plt.show()

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    #plt.imshow(x)
    #plt.show()
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()


    print('Size of x: {}'.format(x.shape))
    x = torch.from_numpy(x).permute(2, 0, 1)
    t = time.time()
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    print ('XX Size: {}'.format(xx.shape))
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    elapsed = time.time() - t
    print ('Forward pass time: {}'.format(elapsed))
    from data import KITTI_CLASSES as labels
    top_k=10

    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib

    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.45:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
    plt.show()