
import cv2
import numpy as np
import pcl
from matplotlib import pyplot as plt
import os
import torch
from torch.autograd import Variable

from ssd_pytorch.data import VOCDetection, VOC_ROOT, VOCAnnotationTransform, KITTIDetection, KITTI_ROOT, KITTIAnnotationTransform
from ssd_pytorch.data import BaseTransform, KITTI_CLASSES as labelmap  # VOC_CLASSES
from ssd_pytorch.ssd import build_ssd

from pointnet_pytorch.My_datasets import PartDataset
from pointnet_pytorch.pointnet import PointNetDenseCls

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def readCalibration(calib_dir, img_idx):

    label_filename = "{}/{}.txt".format(calib_dir, img_idx)
    lines = [line.rstrip() for line in open(label_filename)]

    for line in lines:
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        if(data[0] == 'Tr_velo_to_cam:'):
            Tr_velo_to_cam = np.matrix([[data[1], data[2], data[3], data[4]],
                                        [data[5], data[6], data[7], data[8]],
                                        [data[9], data[10], data[11], data[12]],
                                        [0.0, 0.0, 0.0, 1.0]])
        if(data[0] == 'R0_rect:'):
            R0_rect = np.matrix([[data[1], data[2], data[3], 0.0],
                                 [data[4], data[5], data[6], 0.0],
                                 [data[7], data[8], data[9], 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        if(data[0] == 'P2:'):
            P2 = np.matrix([[data[1], data[2], data[3], data[4]],
                            [data[5], data[6], data[7], data[8]],
                            [data[9], data[10], data[11], data[12]]])

    return Tr_velo_to_cam, R0_rect, P2


def readDetections(label_dir, img_idx):
    label_filename = "{}/{}.txt".format(label_dir, img_idx)
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def readLabels(label_dir, img_idx):
    label_filename = "{}/{}.txt".format(label_dir, img_idx)
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d_GT(line) for line in lines]
    return objects


def read3dBoxes(label_dir, img_idx):
    if img_idx == '000000':
        pcd_id = '0'
    else:
        pcd_id = img_idx.lstrip('0')

    label_filename = "{}/bbox_{}.txt".format(label_dir, pcd_id)
    lines = [line.rstrip() for line in open(label_filename)]
    boxes = [bbox3d(line) for line in lines]
    return boxes


def readPoints(label_dir, img_idx):
    label_filename = "{}/{}.txt".format(label_dir, img_idx)
    lines = [line.rstrip() for line in open(label_filename)]
    points = [Point3D(line) for line in lines]
    return points


def SegmentPoints(points, boxes, class_to_ind, img_idx):

    filename = './PCD_Files2/Labeled/' + \
        'ex{}.txt'.format(img_idx)
    f = open(filename, 'w')

    filtered_points_array = []
    # Check if the Point is inside of the 3D Box
    for ind in range(len(points)):  # range(0,points.shape[0]):
        point = points[ind]
        for index in range(len(boxes)):
            box = boxes[index]
            if ((box.xmin < point.x < box.xmax) and (box.ymin < point.y < box.ymax) and (box.zmin < point.z < box.zmax)):

                filtered_points_array.append(point.coor)
                if box.type == 'Van':
                    f.write('{:.4f} {:.4f} {:.4f} {} '.format(point.coor[0], point.coor[1],
                                                              point.coor[2], class_to_ind['Car']))  # points_array[3]

                elif box.type == 'Person_sitting':
                    f.write('{:.4f} {:.4f} {:.4f} {} '.format(point.coor[0], point.coor[1],
                                                              point.coor[2], class_to_ind['Pedestrian']))  # points_array[3]

                else:
                    f.write('{:.4f} {:.4f} {:.4f} {} '.format(point.coor[0], point.coor[1],
                                                              point.coor[2], class_to_ind[box.type]))
                f.write('{} '.format(point.h))
                f.write('{} '.format(point.w))
                f.write('{} '.format(point.l))
                f.write('{} '.format(point.ry))
                f.write('{:.2f} {:.2f} {:.2f} \n'.format(
                    point.t[0], point.t[1], point.t[2]))
                break
    f.close()


def composeFrustrum(objects, calib_dir, pcd_dir, img_real_id):
    if img_real_id == '000000':
        pcd_id = '0'
    else:
        pcd_id = img_real_id.lstrip('0')  # without 0
    cloud = pcl.load(os.path.join(pcd_dir, '{0}.pcd'.format(pcd_id)))
    points_array = np.asarray(cloud)

    Tr_velo_to_cam, R0_rect, P2 = readCalibration(calib_dir, img_real_id)

    filtered_points_array = [[] for x in range(len(objects))]

    for ind in range(0, points_array.shape[0]):
        y = np.matrix([[points_array[ind][0]], [points_array[ind][1]], [
                      points_array[ind][2]], [1.0]])
        Tr_y = Tr_velo_to_cam*y
        if Tr_y[2] > 0:
            X = P2 * R0_rect * Tr_y
            # For all objects
            for index in range(len(objects)):
                obj = objects[index]
                if ((obj.xmin < X[0]/X[2] < obj.xmax) and (obj.ymin < X[1]/X[2] < obj.ymax)):
                    filtered_points_array[index].append(points_array[ind])

    filtered_points_array = [x for x in filtered_points_array if x != []]
    flat_list = [item for sublist in filtered_points_array for item in sublist]

    return filtered_points_array, flat_list


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line, data=None):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...

        # extract 2d bounding box in 0-based coordinates
        self.xmin = int(data[1])  # left
        self.ymin = int(data[2])  # top
        self.xmax = int(data[3])  # right
        self.ymax = int(data[4])  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

class Object3d_GT(object):
    ''' 3d object GT label '''

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


class Object3d_detected(object):
    ''' 3d object detected label '''

    def __init__(self, label, data):
        data[0:] = [float(x) for x in data[0:]]

        # extract label, truncation, occlusion
        self.type = label  # 'Car', 'Pedestrian', ...

        # extract 2d bounding box in 0-based coordinates
        self.xmin = int(data[0])  # left
        self.ymin = int(data[1])  # top
        self.xmax = int(data[2])  # right
        self.ymax = int(data[3])  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

class Point3D(object):
    ''' 3d point '''

    def __init__(self, label_file_line, data=None):
        data = label_file_line.split(' ')
        data = [float(x) for x in data]

        # extract 2d bounding box in 0-based coordinates
        self.x = float(data[0])  # left
        self.y = float(data[1])  # top
        self.z = float(data[2])  # right

        self.coor = np.array([self.x, self.y, self.z])
        # extract 3d bounding box information
        self.h = data[3]  # box height
        self.w = data[4]  # box width
        self.l = data[5]  # box length (in meters)
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = data[6]
        # location (x,y,z) in camera coord.
        self.t = (data[7], data[8], data[9])


class bbox3d(object):
    ''' 3d object label '''
    # format{minx, maxx, miny, maxy, minz, maxz, label}

    def __init__(self, label_file_line, data=None):
        data = label_file_line.split(',')
        data[0:6] = [float(x) for x in data[0:6]]

        # extract label, truncation, occlusion
        self.type = data[6]  # 'Car', 'Pedestrian', ...
        # self.truncation = data[1] # truncated pixel ratio [0..1]
        # self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        # self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = float(data[0])  # left
        self.xmax = float(data[1])  # right
        self.ymin = float(data[2])  # top
        self.ymax = float(data[3])  # bottom
        self.zmin = float(data[4])  # top
        self.zmax = float(data[5])  # bottom
        self.box3d = np.array(
            [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax])


def ObjectDetection3D(img_idx):

    datatype = 'val'  # 'train' #val
    KITTI_CLASSES = ('Index0', 'Background', 'Cyclist', 'Car', 'Pedestrian')
    class_to_ind = dict(zip(KITTI_CLASSES, range(len(KITTI_CLASSES))))
    # Load the Dataset
    testset = KITTIDetection(
        KITTI_ROOT, [datatype], None, KITTIAnnotationTransform)
    testset3D = PartDataset(
        root="/home/dllab/kitti_object/data_object_image_2", image_sets=[datatype], train=False)
    # Necessary Directories
    root_dir = "/home/dllab/kitti_object/data_object_image_2"
    pcd_dir = "/home/dllab/kitti_object/data_object_velodyne/pcl"
    segmented_pcd_dir = "./PCD_Files1"
    data_set = "training"
    images_dir = os.path.join(root_dir, data_set, "image_{0}".format(2))
    detection_dir = './Detections'
    calib_dir = '/home/dllab/kitti_object/data_object_velodyne/data_object_calib/training/calib'
    box_dir = './bbox_labels_new'
    points_dir = "./PCD_Files2/DetectionLocations"
    label_dir = os.path.join(root_dir, data_set, "label_{0}".format(2))
    network3D = './Pointnet/seg/seg_model_24.pth'
    saved_PCD_dir = './Final'
    img_real_id = testset.img_id_return(img_idx)
    print('indx = {} Image: {}'.format(img_idx, img_real_id))

    pcd_id = img_real_id.lstrip('0')

    objects_GT = readLabels(label_dir, img_real_id)

    # Read the boxes and Points
    boxes = read3dBoxes(box_dir, img_real_id)
    points = readPoints(points_dir, img_real_id)

    # 2D Box Detection

    net = build_ssd('test', 300, 4)    # initialize SSD
    # net.load_state_dict(torch.load(args.weights))
    net.load_weights(
        './ssd_pytorch/weights/ssd300_Resz_KITTI_105000.pth')

    image = testset.pull_image(img_idx)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300)).astype(np.float32)

    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()

    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    print('XX Size: {}'.format(xx.shape))
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    # plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib

    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    objs = []
    for i in range(detections.size(1)):
        j = 0
        if i == 2:
            limit = 0.5
        else:
            limit = 0.1
        while detections[0, i, j, 0] >= limit:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(image,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(image, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
            objs.append(Object3d_detected(labelmap[i - 1], pt))
            j += 1

    cv2.imshow('frame', image)
    cv2.waitKey(1) & 0xFF
    objects = objs

    # Get Frustrum
    frustrum_points, frustrum_points_total = composeFrustrum(
        objects, calib_dir, pcd_dir, img_real_id)
    frustrumcloud = pcl.PointCloud(
        np.array(frustrum_points_total, dtype=np.float32))
    pcl.save(frustrumcloud, "{}/frustrum_{}.pcd".format(saved_PCD_dir, pcd_id))

    # Segment Points
    # Apply 3D segmentation network
    classifier = PointNetDenseCls(k=4)
    classifier.load_state_dict(torch.load(network3D))
    classifier.eval()

    second_index = 0
    str_parser = './Final/frustrum_{0}.pcd '.format(
        pcd_id)
    for ind in range(len(frustrum_points)):
        obj_points = frustrum_points[ind]

        obj_points_num = np.asarray(obj_points)
        choice = np.random.choice(len(obj_points_num), 2500, replace=True)
        point_set = obj_points_num[choice, :]
        point_nn = point_set
        point_set = point_set / np.absolute(point_set).max(axis=0)

        point = torch.from_numpy(point_set)
        point = point.transpose(1, 0).contiguous()

        point = Variable(point.view(1, point.size()[0], point.size()[1]))

        pred, _ = classifier(point)
        pred_choice = pred.data.max(2)[1]
        np.set_printoptions(threshold=np.nan)

        obj_index = np.nonzero(pred_choice.numpy()[0])
        pred_points = point_nn[obj_index]
        if pred_points.size != 0:
            predcloud = pcl.PointCloud(np.array(pred_points, dtype=np.float32))
            pcl.save(
                predcloud, "{}/segmented_{}_{}.pcd".format(saved_PCD_dir, pcd_id, ind))
            str_parser = str_parser + \
                "{}/segmented_{}_{}.pcd ".format(saved_PCD_dir,
                                                 pcd_id, second_index)
            second_index = second_index + 1
    print(str_parser)
    os.system("./Visualizers/showObjects/build/showObjects {} {}".format(pcd_id, second_index))

    # Draw 3D Boxes
    os.system("./Visualizers/showBoxes/build/showBoxes {} {}".format(pcd_id, second_index))
    #os.system('pcl_viewer {0} &'.format(str_parser))
    #os.system('pcl_viewer ./Final/frustrum_{0}.pcd ./Final/segmented_{0}.pcd &'.format(pcd_id))
    #os.system("/home/emeka/Schreibtisch/AIS/deleteme/build/test_pcl {}".format(pcd_id))

if __name__ == "__main__":
    # indx = 120 Image: 000263
    ObjectDetection3D(120)
