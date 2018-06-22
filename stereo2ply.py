import cv2
import numpy as np
import math
import utils
from plyfile import PlyData, PlyElement
import subprocess
import os
import json
pi = 3.14159265
size0 = 512

def xyz2pc (XYZ, mask, min_z=0, max_z=10000):
    pcdt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    fa = np.empty(len(mask[mask>0]), dtype=pcdt)
    i = 0
    for x in range(0, XYZ.shape[1]):
        for y in range(0, XYZ.shape[0]):
            if (mask[y,x]==1):
                fa[i] = np.array([(XYZ[y,x,0], XYZ[y,x,1], XYZ[y,x,2])], dtype=pcdt)
                i = i + 1
    return fa

def call_PSMNet(imgL_path, imgR_path, disp_path, maxdisp = 192, imsize = 1024, mirror=0):
    # Calling PSMNet (https://github.com/JiaRenChang/PSMNet) to reconstruct stereo disparity map

    command = ['python2', 'PSMNet/PSMNet_stereo.py', 
               '--maxdisp', str(maxdisp), 
               '--loadmodel', 'PSMNet/models/pretrained_model_KITTI2015.tar', 
               '--imgL', imgL_path, 
               '--imgR', imgR_path, 
               '--out', disp_path, 
               '--imsize', str(imsize),
               '--mirror', str(mirror)]
    subprocess.call(command)
    
def run_openpose(imdir, outdir):
    command = ['/home/vaheta/builds/openpose/build/examples/openpose/openpose.bin', 
            '--image_dir', imdir, 
            '--face', 
            '--net_resolution', '656x368', 
            '--write_json', outdir,
            '--write_images', outdir,
            '--display', '0']
    subprocess.call(command)
    
def read_keypoint_json(jsonpath):
    with open(jsonpath) as f:
        data = json.load(f)
        facedict = {'x':[], 'y':[], 'c':[]}
        for i,k in enumerate(data['people'][0]['face_keypoints_2d']):
            if i%3==0:
                facedict['x'].append(k)
            elif i%3==1:
                facedict['y'].append(k)
            elif i%3==2:
                facedict['c'].append(k)
    return facedict

### RECTIFYING IMAGES AND SAVING RESULTING IMAGES

calp = np.load ('26.05/calib_lr.npz')
lFrame = cv2.imread('26.05/for_rectification/left.png')
rFrame = cv2.imread('26.05/for_rectification/right.png')

img_shape = lFrame[:,:,0].shape[::-1]

(RCT1, RCT2, P1, P2, dsp2dm, ROI1, ROI2) = cv2.stereoRectify(
            calp['M1'], calp['d1'],
            calp['M2'], calp['d2'],
            img_shape, calp['R'], calp['T'],
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, alpha=-1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        calp['M1'], calp['d1'], RCT1,
        P1, img_shape, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        calp['M2'], calp['d2'], RCT2,
        P2, img_shape, cv2.CV_32FC1)

fixedLeft = cv2.remap(lFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
fixedRight = cv2.remap(rFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)

imgL_path = 'rectified/rect_l.png'
imgR_path = 'rectified/rect_r.png'

fixedLeft = cv2.resize(fixedLeft, (size0,size0))
fixedRight = cv2.resize(fixedRight, (size0,size0))

cv2.imwrite(imgL_path, fixedLeft)
cv2.imwrite(imgR_path, fixedRight)

### CALCULATING DISPARITY MAP

disp_path = 'rectified/dispmap.npy'

call_PSMNet(imgL_path, imgR_path, disp_path, imsize=size0)

dispmap = np.load(disp_path)

### TRANSLATING DISPARITY TO DEPTH MAP

dispmap = cv2.resize(dispmap, img_shape)

dispmap = dispmap*img_shape[0]/size0

XYZ = cv2.reprojectImageTo3D(dispmap, dsp2dm)

### CREATING FACE MASK FOR DEPTH MAP FILTERING

imdir = 'rectified/'
run_openpose(imdir, imdir+'openpose_out/')
filename, extension = os.path.splitext(os.path.basename(imgL_path))
facepoints = read_keypoint_json((imdir+'openpose_out/'+filename+'_keypoints.json'))
facepoints['x'] = [x * img_shape[0]/size0 for x in facepoints['x']]
facepoints['y'] = [x * img_shape[1]/size0 for x in facepoints['y']]
facepoints['z'] = len(facepoints['x']) * [0]
for i in range(0, len(facepoints['x'])):
    facepoints['z'][i] = XYZ[int(facepoints['y'][i]), int(facepoints['x'][i]), 2]

min_x = int(0.9*min(facepoints['x']))
max_x = int(1.1*max(facepoints['x']))
min_y = min(facepoints['y'])
max_y = int(1.1*max(facepoints['y']))
min_y = int(min_y-0.5*(max_y-min_y))
min_z = min(facepoints['z'])-200
max_z = max(facepoints['z'])+200

mask = np.zeros(img_shape[::-1])

mask[min_y:max_y, min_x:max_x] = 1
mask[XYZ[:,:,2]>max_z] = 0
mask[XYZ[:,:,2]<min_z] = 0

### TRANSLATING DEPTH MAP TO POINT CLOUD AND SAVING TO PLY FILE

fa = xyz2pc(XYZ, mask)

el = PlyElement.describe(fa, 'vertex')

ply_file_path = 'rectified/respc.ply'
PlyData([el], text=True).write(ply_file_path)