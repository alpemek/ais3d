from __future__ import print_function
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
parser.add_argument('--weights', default='/home/emeka/Schreibtisch/AIS/ais3d/ssd.pytorch-master/weights/ssd300_KITTI_5000.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = cv2.resize(frame, (300, 300)).astype(np.float32) #

        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        #plt.figure(figsize=(10,10))
        #plt.imshow(x)
        #plt.show()
        #x -= (104.0, 117.0, 123.0)#
        x = x.astype(np.float32)#
        x = x[:, :, ::-1].copy()#
        #print('Size of x: {}'.format(x.shape))
        #x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = torch.from_numpy(x).permute(2, 0, 1) #frame -> x
        xx = Variable(x.unsqueeze(0))
        print ('XX Size: {}'.format(xx.shape))

        if torch.cuda.is_available():
            xx = xx.cuda()
        t = time.time()
        y = net(xx)  # forward pass

        elapsed = time.time() - t
        print ('Forward pass time: {}'.format(elapsed))
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.4:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    #stream = WebcamVideoStream(src=0).start()  # default camera
    #this line makes stream from a file saved on harddrive
    stream = FileVideoStream('/home/emeka/Downloads/driving/driving.avi').start()
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        #this for loop was inserted in order to try to make the live in real time (which is currently not happening)
        for i in range(25):
            frame = stream.read()

            key = cv2.waitKey(1) & 0xFF
            # update FPS counter
            fps.update()

        t = time.time()
        frame = predict(frame)
        print ('Frame Size: {}'.format(frame.shape))
        elapsed = time.time() - t
        print (elapsed)
        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, KITTI_CLASSES as labelmap #VOC_CLASSES
    from ssd import build_ssd

    net = build_ssd('test', 300, 4)    # initialize SSD
    #net.load_state_dict(torch.load(args.weights))
    net.load_weights('../weights/ssd300_KITTI_110000.pth')
    transform = BaseTransform(net.size, (104/2048.0, 117/2048.0, 123/2048.0))
    net.cuda()
    fps = FPS().start()
    cv2_demo(net, transform) #.eval()
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
