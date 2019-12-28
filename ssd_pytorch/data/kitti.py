import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

KITTI_CLASSES = (  # always index 0
    'Cyclist', 'Car', 'Pedestrian')

# note: if you used our download scripts, this should be right
KITTI_ROOT = osp.join("/home/dllab/kitti_object/data_object_image_2")


class KITTIAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(KITTI_CLASSES, range(len(KITTI_CLASSES))))
       # self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        objects = [Object3d(line) for line in target]

        for obj in objects:
            name = obj.type
            bbox = []

            bbox.append(obj.xmin / width)
            bbox.append(obj.ymin / height)
            bbox.append(obj.xmax / width)
            bbox.append(obj.ymax / height)
            try:
                label_idx = self.class_to_ind[name]
            except KeyError:
                continue

            bbox.append(label_idx)
            res += [bbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class KITTIDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('trainval')],
                 transform=None, target_transform=KITTIAnnotationTransform(),
                 dataset_name='KITTI'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        images_dir = osp.join(KITTI_ROOT, "training", "image_{0}".format(2))
        label_dir = osp.join(KITTI_ROOT, "training", "label_{0}".format(2))
        self._annopath = label_dir  # osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = images_dir  # osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (name) in image_sets:
            rootpath = osp.join(self.root)
            for line in open(osp.join(rootpath, name + '.txt')):
                self.ids.append((line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def img_id_return(self, index):
        return self.ids[index]

    def pull_item(self, index):
        img_id = self.ids[index]

        image_dir = "{}/{}.png".format(self._imgpath, img_id)
        label_filename = "{}/{}.txt".format(self._annopath, img_id)
        target = [line.rstrip() for line in open(label_filename)]

        img = cv2.imread(image_dir)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread("{}/{}.png".format(self._imgpath, img_id), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        label_filename = "{}/{}.txt".format(self._annopath, img_id)
        anno = [line.rstrip() for line in open(label_filename)]
        gt = self.target_transform(anno, 1, 1)
        return str(img_id), gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


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
