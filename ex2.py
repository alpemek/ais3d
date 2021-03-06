'''
University of Freiburg - Laboratory, Deep Learning for Autonomous Driving
Project 2- Exercise 2
Alp Emek
Marcelo Chulek

The code is converted to python from the matlab code provided in KITTI dataset.
16.5.2018
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

root_dir = "/home/dllab/kitti_object/data_object_image_2"
data_set = "training"

# get sub-directories
cam = 2  # 2 = left color camera
images_dir = os.path.join(root_dir, data_set, "image_{0}".format(cam))
label_dir = os.path.join(root_dir, data_set, "label_{0}".format(cam))
calib_dir = os.path.join(root_dir, data_set, "calib")

def readCalibration(calib_dir, img_idx, cam):
    lines = [line.rstrip() for line in open(
        "{}/{:06d}.txt".format(calib_dir, img_idx))]
    P = [line.split(' ') for line in lines]
    P = P[cam]
    P.pop(0)
    P = np.asarray(P)
    P = P.reshape(3, -1)
    return P


def readLabels(label_dir, img_idx):
    label_filename = "{}/{:06d}.txt".format(label_dir, img_idx)
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line, data=None):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.occlusion = int(data[2])
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = int(data[4])  # left
        self.ymin = int(data[5])  # top
        self.xmax = int(data[6])  # right
        self.ymax = int(data[7])  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = data[14]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' %
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' %
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' %
              (self.t[0], self.t[1], self.t[2], self.ry))


def drawBox2D(img, obj):
    cv2.rectangle(img, (obj.xmin, obj.ymin),
                  (obj.xmax, obj.ymax), (0, 255, 0), 2)

img_idx = 4  # Index of the image to be shown
image_dir = "{}/{:06d}.png".format(images_dir, img_idx)
P = readCalibration(calib_dir, img_idx, cam)
objects = readLabels(label_dir, img_idx)

# Show image

image = cv2.imread(image_dir, 1)

# plot rect for every box
for objind in range(len(objects)):
    drawBox2D(image, objects[objind])

cv2.imshow("test image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
