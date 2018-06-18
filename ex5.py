from __future__ import print_function

import numpy as np
import pcl
import numpy as np
import os.path

#/home/dllab/kitti_object/data_object_velodyne/data_object_calib/training/calib
def readCalibration(calib_dir,img_idx):

  label_filename = "{}/{:06d}.txt".format(calib_dir,img_idx)
  lines = [line.rstrip() for line in open(label_filename)]

  for line in lines:
      data = line.split(' ')
      data[1:] = [float(x) for x in data[1:]]
      if(data[0] == 'Tr_velo_to_cam:'):
        #rot [0][0]data [1]
        print('a')
        Tr_velo_to_cam = np.matrix([[data[1],data[2],data[3],data[4]],[data[5],data[6], data[7], data[8]],[data[9],data[10],data[11], data[12]],[0.0,0.0,0.0,1.0]])
      if(data[0] == 'R0_rect:'):
        R0_rect = np.matrix([[data[1],data[2],data[3],0.0],[data[4],data[5], data[6],0.0],[data[7],data[8],data[9],0.0],[0.0,0.0,0.0,1.0]])
      if(data[0] == 'P2:'):
        P2 = np.matrix([[data[1],data[2],data[3],data[4]],[data[5],data[6],data[7],data[8]],[data[9],data[10],data[11], data[12]]])


  return Tr_velo_to_cam, R0_rect, P2

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
    self.l = data[10]  # box length (in meters)
    self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
    self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

if __name__ == "__main__":
    #LOADING THE CLOUD
    cloud = pcl.load('/home/dllab/kitti_object/data_object_velodyne/pcl/3.pcd')
    points_array = np.asarray(cloud)
    #print(a[2][1])



    #READING THE DATA FROM THE FILE
    root_dir="/home/dllab/kitti_object/data_object_image_2"
    data_set = "training"
    img_idx=3;
    images_dir = os.path.join(root_dir,data_set, "image_{0}".format(2));
    label_dir = os.path.join(root_dir,data_set, "label_{0}".format(2));
    calib_dir = '/home/dllab/kitti_object/data_object_velodyne/data_object_calib/training/calib'
    image_dir = "{}/{:06d}.png".format(images_dir,img_idx);
    Tr_velo_to_cam, R0_rect, P2 = readCalibration(calib_dir,img_idx);
    objects = readLabels(label_dir,img_idx);
    filtered_points_array = []
    #DOT PRODUCT
    #Tr_velo_to_cam * y (4x4) (4x1)
    for ind in range(0,points_array.shape[0]):
        y = np.matrix([[points_array[ind][0]],[points_array[ind][1]],[points_array[ind][2]],[1.0]]);
        Tr_y = Tr_velo_to_cam*y
        if Tr_y[2] > 0:
            X = P2 * R0_rect * Tr_y
            #print(X[1])
            obj=objects[0];
            if ( ( obj.xmin < X[0]/X[2] < obj.xmax ) and ( obj.ymin < X[1]/X[2] < obj.ymax ) ):
                print(X)
                filtered_points_array.append(points_array[ind])



       # print(Tr_y)
  #  for x in np.nditer(a):
   #     print(x)
        # 2d equal ret times ba ba times tr velo times y


    #outcloud = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5],[6,7,8]], dtype=np.float32))
    outcloud = pcl.PointCloud(np.array(filtered_points_array, dtype=np.float32))
    pcl.save(outcloud, "test.pcd")







