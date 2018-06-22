from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--imsize', type=int, default=512,
                    help='resized image size')
parser.add_argument('--imgL', default=None,
                    help='path to left image')
parser.add_argument('--imgR', default=None,
                    help='path to right image')
parser.add_argument('--out', default=None,
                    help='path to output image')
parser.add_argument('--mirror', type=int, default=0,
                    help='enables mirroring image to switch between left-right')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
# model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR = Variable(torch.FloatTensor(imgL)), Variable(torch.FloatTensor(imgR))

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)

    test_left_img = args.imgL
    test_right_img = args.imgR

    newsize = args.imsize
    
    imgL_o = skimage.io.imread(test_left_img)
    imgR_o = skimage.io.imread(test_right_img)
    imgL_o = skimage.transform.resize(imgL_o, (newsize, newsize)).astype('float32')
    imgR_o = skimage.transform.resize(imgR_o, (newsize, newsize)).astype('float32')
    if args.mirror:
        imgL_o = imgL_o[:,::-1,:].copy()
        imgR_o = imgR_o[:,::-1,:].copy()
    # skimage.io.imsave('testim1.png', imgL_o[:,:,0:3])
    # skimage.io.imsave('testim2.png', imgR_o[:,:,0:3])
    imgL = processed(imgL_o[:,:,0:3]).numpy()
    imgR = processed(imgR_o[:,:,0:3]).numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to (newsize, newsize)
    top_pad = newsize-imgL.shape[2]
    left_pad = newsize-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    start_time = time.time()
    img = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))
    if args.mirror:
        img = img[:,::-1]
    np.save(args.out, img)
    # skimage.io.imsave(args.out,(img*256).astype('uint16'))

if __name__ == '__main__':
   main()






