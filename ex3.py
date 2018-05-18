'''
University of Freiburg - Laboratory, Deep Learning for Autonomous Driving
Project 2- Exercise 2
Alp Emek
Marcelo Chulek

This code is UNDER DEVELOPMENT. IT IS SUPPOSED TO RESIZE THE KITTI database by a factor of 0.5 and save it on the training_resized directory
16.5.2018
'''
from __future__ import print_function
import torch
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import os.path



root_dir="/home/dllab/kitti_object/data_object_image_2"
data_set = "training"

# get sub-directories
cam = 2; # 2 = left color camera
images_dir = os.path.join(root_dir,data_set, "image_{0}".format(cam));
print(images_dir)
label_dir = os.path.join(root_dir,data_set, "label_{0}".format(cam));
calib_dir = os.path.join(root_dir,data_set, "calib");
print()

def readCalibration(calib_dir,img_idx,cam):
  #P=np.fromfile("{}/{:06d}.txt".format(calib_dir,img_idx),dtype=float, count=-1, sep=" ")
  #P=P[cam,:]
  #P=P.reshape(3,-1)

  lines = [line.rstrip() for line in open("{}/{:06d}.txt".format(calib_dir,img_idx))]
  P = [line.split(' ') for line in lines]
  P=P[cam]
  P.pop(0)
  P=np.asarray(P)
  P=P.reshape(3,-1)
  return P

def readLabels(label_dir,img_idx):
  label_filename = "{}/{:06d}.txt".format(label_dir,img_idx)
  lines = [line.rstrip() for line in open(label_filename)]
  objects = [Object3d(line) for line in lines]
  return objects

class Object3d(object):
  ''' 3d object label '''
  def __init__(self, label_file_line, data=None):
    data = label_file_line.split(' ')
    data[1:] = [float(x) for x in data[1:]]

    # extract label, truncation, occlusion
    self.type = data[0] # 'Car', 'Pedestrian', ...
    self.truncation = data[1] # truncated pixel ratio [0..1]
    self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
    self.alpha = data[3] # object observation angle [-pi..pi]

    # extract 2d bounding box in 0-based coordinates
    self.xmin = int(data[4]) # left
    self.ymin = int(data[5]) # top
    self.xmax = int(data[6]) # right
    self.ymax = int(data[7]) # bottom
    self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

    # extract 3d bounding box information
    self.h = data[8] # box height
    self.w = data[9] # box width
    self.l = data[10] # box length (in meters)
    self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
    self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

  def print_object(self):
    print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
        (self.type, self.truncation, self.occlusion, self.alpha))
    print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
        (self.xmin, self.ymin, self.xmax, self.ymax))
    print('3d bbox h,w,l: %f, %f, %f' % \
        (self.h, self.w, self.l))
    print('3d bbox location, ry: (%f, %f, %f), %f' % \
        (self.t[0],self.t[1],self.t[2],self.ry))

def drawBox2D(img,obj):
  #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
  cv2.rectangle(img,(obj.xmin,obj.ymin),(obj.xmax,obj.ymax),(0,255,0),2)
  #cv2.rectangle(img,(384,0),(510,128),(0,255,0),30)


img_idx=2; # Index of the image to be shown
image_dir = "{}/{:06d}.png".format(images_dir,img_idx);
P = readCalibration(calib_dir,img_idx,cam);
objects = readLabels(label_dir,img_idx);

#Show image

image = cv2.imread(image_dir, 1)
#image = np.zeros((512,512,3), np.uint8)
#cv2.imshow("test image", image)
#cv2.rectangle(image,(384,0),(510,128),(0,255,0),30)
#cv2.imshow("test image", image)
#plot rect for every box
for objind in range(len(objects)):
  drawBox2D(image,objects[objind]);
  #[corners,face_idx] = computeBox3D(objects(obj_idx),P);
  #orientation = computeOrientation3D(objects(obj_idx),P);
  #drawBox3D(h, objects(obj_idx),corners,face_idx,orientation);
cv2.imshow("test image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()