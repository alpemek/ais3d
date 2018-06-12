from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='KITTI', choices=['VOC', 'COCO', 'KITTI'],
                    type=str, help='VOC ,COCO or KITTI')
parser.add_argument('--dataset_root', default=KITTI_ROOT,  # default=VOC_ROOT KITTI_ROOT
                    help='Dataset root directory path')
#THIS IS THE ALREADY PRE TRAINED VGG 16 NETWORK
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
#reducing the batch size reduces the memory usage. and u should reduce the learning rate also
#this value was reduced. originally the batch size was 32
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
#parser.add_argument('--resume', default=None, type=str,
#                    help='Checkpoint state_dict file to resume training from')
#IF ONE DOESNT WANT TO RESUME, SET defaul=None
parser.add_argument('--resume', default=None, type=str, #default='/home/emeka/Schreibtisch/AIS/ais3d/ssd.pytorch-master/weights/ssd300_KITTI_75000.pth'
                    help='Checkpoint state_dict file to resume training from')
#parser.add_argument('--start_iter', default=0, type=int,
#                    help='Resume training at this iter')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
#HERE CUDA IS FORCED TO BE THE STANDARD DEVICE
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
#THIS LEARNING RATE SHOULD IDEALLY CHANGE WITH THE BATCH SIZE. SMALLER BATCH, SMALLER LEARNING RATE
parser.add_argument('--lr', '--learning-rate', default=0.1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("CUDA device loaded successfully")
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    #THIS LINE IS WHERE ONE SPECIFIES THE DATASET. VOC SHOULD BE CHANGED
    if args.dataset == 'VOC':
        #if args.dataset_root == COCO_ROOT:
        #    parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        print("Loading the dataset")
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        print("Dataset loaded")
    if args.dataset == 'KITTI':
        cfg = kitti
        print("Loading the dataset")
        dataset = KITTIDetection(root=args.dataset_root,image_sets=[('train')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        print("Dataset loaded")
    #building the ssd network. WHEN ONE DOES STEP BY STEP, THIS NEXT LINE CONSUMES A LOT OF TIME
    print("Building the SSD Network. This make take a while. Go and grab a coffee at the pool machine...")
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    print("SSD Network was created")
    net = ssd_net
    iter_datasets = len(dataset) // args.batch_size
    epoch_size = cfg['max_iter'] // iter_datasets

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        start_epoch = args.start_iter // iter_datasets
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        start_epoch = 0
    if args.cuda:
        net = net.cuda()
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    #stocastic gradient descent
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')
    print('Length of the dataset: {}'.format(len(dataset)))

   # epoch_size = len(dataset) // args.batch_size

    #print('epoch_size ' + str(epoch_size))
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,num_workers=args.num_workers,shuffle=True, collate_fn=detection_collate,pin_memory=True)
    # create batch iterator
    #batch_iterator = iter(data_loader) #uncomment
    #ADDED

    print('Number of Iterations per Epoch: {}'.format(iter_datasets))
    print('Number of Epochs Left: {}'.format(epoch_size-start_epoch))
    for epoch in range(start_epoch, epoch_size):
        for iteration, (images, targets) in enumerate(data_loader):
            if iteration in cfg['lr_steps']:
                print('cfg[lr_steps]' + cfg['lr_steps'])
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # BACKPROPAGATION LINE
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            if iteration % 10 == 0:
                #print('timer: %.4f sec.' % (t1 - t0))
                print('Epoch: {:} /{:}'.format(epoch+1,epoch_size)  +' iter: {:}/{:} '.format(iteration,iter_datasets) + ' || Loss: %.4f ||' % (loss.data[0]))

            #Saving the intermediate state
            if (iteration) != 0 and ((epoch)*iter_datasets +iteration) % 5000 == 0:
                print('Saving state, Total iter:', (epoch)*iter_datasets +iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_Resz_KITTI_' +
                           repr( (epoch)*iter_datasets +iteration ) + '.pth')
    #here the network is saved
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '_' + repr(cfg['max_iter']) + '.pth')
    print('Network saved')

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__=='__main__':
    train()
