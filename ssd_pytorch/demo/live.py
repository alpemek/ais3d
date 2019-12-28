import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream, FileVideoStream
import argparse
import numpy as np
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/home/emeka/Schreibtisch/AIS/ais3d/ssd_pytorch_master/weights/ssd300_Resz_KITTI_105000.pth',  # ssd300_Resz_KITTI_105000
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = cv2.resize(frame, (300, 300)).astype(np.float32)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)  # frame -> x
        xx = Variable(x.unsqueeze(0))

        if torch.cuda.is_available():
            xx = xx.cuda()
        t = time.time()
        y = net(xx)  # forward pass

        elapsed = time.time() - t
        detections = y.data
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            if i == 2:
                limit = 0.4
            else:
                limit = 0.05
            while detections[0, i, j, 0] >= limit:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    print("[INFO] starting threaded video stream...")
    ids = list()
    for line in open('/home/emeka/Schreibtisch/AIS/ais3d/' + 'FreiburgImages.txt'):
        ids.append((line.strip()))
    img_id = 1

    while True:
        # grab next frame
        frame = cv2.imread(ids[img_id], cv2.IMREAD_COLOR)
        img_id += 1
        frame = predict(frame)
        
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1500, 1100)
        cv2.imshow('frame', frame)
        cv2.waitKey(1) & 0xFF

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, KITTI_CLASSES as labelmap  # VOC_CLASSES
    from ssd import build_ssd

    net = build_ssd('test', 300, 4)    # initialize SSD
    net.load_weights(args.weights)
    transform = BaseTransform(net.size, (104/2048.0, 117/2048.0, 123/2048.0))
    net.cuda()
    cv2_demo(net, transform) 
    cv2.destroyAllWindows()

