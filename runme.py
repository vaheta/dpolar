import cv2
import numpy as np
import math
import utils
from plyfile import PlyData, PlyElement
import subprocess
import matplotlib.pyplot as plt
import os
import json
import statistics
%matplotlib notebook

# Константы и параметры
pi = 3.14159265
size0 = 1024
maxdisp = 192
n = 1.6
xvals = np.mat([pi/4, pi/2, 0])


def xyz2pc (XYZ, mask):
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
    # Вызываем PSMNet 

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
    
def read_keypoint_json(jsonpath, rotate=False, ymax = 0):
    with open(jsonpath) as f:
        data = json.load(f)
        facedict = {'x':[], 'y':[], 'c':[]}
        # детектируем корректного человека
        j0 = 0
        for j in range(0, len(data['people'])):
            csum = 0
            for i,k in enumerate(data['people'][j]['face_keypoints_2d']):
                if i%3==2:
                    csum = csum + k/(len(data['people'][j]['face_keypoints_2d'])/3)
            if csum > 0.5:
                j0 = j
        for i,k in enumerate(data['people'][j0]['face_keypoints_2d']):
            if rotate:
                if i%3==0:
                    facedict['y'].append(k)
                elif i%3==1:
                    facedict['x'].append(ymax-k-1)
                elif i%3==2:
                    facedict['c'].append(k)
            else:
                if i%3==0:
                    facedict['x'].append(k)
                elif i%3==1:
                    facedict['y'].append(k)
                elif i%3==2:
                    facedict['c'].append(k)
    return facedict

def color_correction(im, a):
    ymax = im.shape[0]
    xmax = im.shape[1]
    imout = np.zeros(im.shape)
    for y in range(0,ymax):
        for x in range(0,xmax):
            r = int(im[y,x,0])
            g = int(im[y,x,1])
            b = int(im[y,x,2])
            vec = np.array([r,g,b,r**2,g**2,b**2,r*g,r*b,b*g,r**3,g**3,b**3,(r**2)*g,r*(g**2),(r**2)*b,r*(b**2),(b**2)*g,b*(g**2)])
            rgbnew = np.dot(vec,a.T)
            imout[y,x,:] = np.around(rgbnew)
    return imout

def stacker(lFrame, rFrame, calp, rotate=False):
    
    ### Ректификация
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

#     if rotate:
#         fixedLeft = np.flipud(np.rot90(fixedLeft,3))
#         fixedRight = np.flipud(np.rot90(fixedRight,3))
#         img_shape = fixedRight[:,:,0].shape[::-1]
    
    imgL_path = 'rectified/rect_l.png'
    imgR_path = 'rectified/rect_r.png'
    
    cv2.imwrite(imgL_path, fixedLeft)
    cv2.imwrite(imgR_path, fixedRight)
    
    imdir = 'rectified/'
    run_openpose(imdir, imdir+'openpose_out/')

    filenamel, extensionl = os.path.splitext(os.path.basename(imgL_path))
    filenamer, extensionr = os.path.splitext(os.path.basename(imgR_path))
    
    if rotate:
        facepoints_l = read_keypoint_json((imdir+'openpose_out/'+filenamel+'_keypoints.json'), True, img_shape[1])
        facepoints_r = read_keypoint_json((imdir+'openpose_out/'+filenamer+'_keypoints.json'), True, img_shape[1])
        fixedLeft = np.rot90(fixedLeft,3)
        fixedRight = np.rot90(fixedRight,3)
        img_shape = fixedRight[:,:,0].shape[::-1]
    else:
        facepoints_l = read_keypoint_json((imdir+'openpose_out/'+filenamel+'_keypoints.json'))
        facepoints_r = read_keypoint_json((imdir+'openpose_out/'+filenamer+'_keypoints.json'))        
    
    diff = [0]*len(facepoints_l['x'])
    for i in range(0, len(facepoints_l['x'])):
        diff[i] = int(facepoints_l['x'][i] - facepoints_r['x'][i])
    dispwindow = max(diff)-min(diff)
    dispadd = maxdisp - 1*dispwindow
    if dispadd < 0:
        print ("ATTENTION! NEED TO TWEEK DISP HACK PARAMETERS")


    if rotate:
        k1 = 0.5
        k2 = 0.1
    else:
        k1 = 0.1
        k2 = 0.5
        
    min_x_lt = min(facepoints_l['x'])
    max_x_lt = max(facepoints_l['x'])
    min_y_lt = min(facepoints_l['y'])
    max_y_lt = max(facepoints_l['y'])

    min_x_l = int(min_x_lt - 0.1*(max_x_lt-min_x_lt))
    max_x_l = int(max_x_lt + k1*(max_x_lt-min_x_lt))
    min_y_l = int(min_y_lt - k2*(max_y_lt-min_y_lt))
    max_y_l = int(max_y_lt + 0.1*(max_y_lt-min_y_lt))

    min_x_rt = min(facepoints_r['x'])
    max_x_rt = max(facepoints_r['x'])
    min_y_rt = min(facepoints_r['y'])
    max_y_rt = max(facepoints_r['y'])

    min_x_r = int(min_x_rt - 0.1*(max_x_rt-min_x_rt))
    max_x_r = int(max_x_rt + k1*(max_x_rt-min_x_rt))
    min_y_r = int(min_y_rt - k2*(max_y_rt-min_y_rt))
    max_y_r = int(max_y_rt + 0.1*(max_y_rt-min_y_rt))

    deltadisp = (min_x_l-dispadd) - min_x_r
    
    imwidth = max([max_x_l - (min_x_l-dispadd), (max_x_r+dispadd) - min_x_r])
    
    iml = fixedLeft[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth),:]
    imr = fixedRight[min_y_r:max_y_r, min_x_r:(min_x_r+imwidth),:]

    iml = cv2.resize(iml, (size0,size0))
    imr = cv2.resize(imr, (size0,size0))

    imgL_cut_path = 'rectified/rect_l_cut.png'
    imgR_cut_path = 'rectified/rect_r_cut.png'

    cv2.imwrite(imgL_cut_path, iml)
    cv2.imwrite(imgR_cut_path, imr)
    
    ### Вычисляем карту смещений

    disp_path = 'rectified/tempdispmap.npy'
    call_PSMNet(imgL_cut_path, imgR_cut_path, disp_path, imsize=size0, maxdisp=maxdisp)
    dispmap_load = np.load(disp_path)
    
    ### Преобразовываем карту смещений в карту глубин

    dispmap_cut = cv2.resize(dispmap_load, (imwidth, (max_y_l-min_y_l)))
    dispmap_cut = (dispmap_cut*imwidth/size0) + deltadisp
    dispmap = cv2.resize(dispmap_cut, img_shape)/10000
    dispmap[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth)] = dispmap_cut
    if rotate:
        dispmap = -np.rot90(dispmap)
        img_shape = dispmap.shape[::-1]
    XYZ = cv2.reprojectImageTo3D(dispmap, dsp2dm)

    if rotate:
        dmtemp = np.flipud(np.rot90(XYZ,3))[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth),2]
    else:
        dmtemp = XYZ[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth),2]
    
    laplacian = cv2.Laplacian(dmtemp,cv2.CV_32F)
    sobelx = cv2.Sobel(dmtemp,cv2.CV_32F,1,0,ksize=5)
    sobely = cv2.Sobel(dmtemp,cv2.CV_32F,0,1,ksize=5)

    filtermapx = np.zeros(dmtemp.shape)
    filtermapy = np.zeros(dmtemp.shape)
    filtermap = np.zeros(dmtemp.shape)

    filtermapx[abs(sobelx)<np.mean(abs(sobelx))] = 1
    filtermapy[abs(sobely)<np.mean(abs(sobely))] = 1
    filtermap = filtermapx*filtermapy

    if rotate:
        mask = np.zeros(img_shape, np.uint8)
        mask[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth)] = filtermap
        mask = np.rot90(mask)
    else:
        mask = np.zeros(img_shape[::-1], np.uint8)
        mask[min_y_l:max_y_l, (min_x_l-dispadd):(min_x_l-dispadd+imwidth)] = filtermap
    dispmap = dispmap*mask
    
    if rotate:
        dmrot = np.rot90(XYZ[:,:,2],3)
    else:
        dmrot = XYZ[:,:,2]
    facepoints_l['z'] = len(facepoints_l['x']) * [0]
    for i in range(0, len(facepoints_l['x'])):
        facepoints_l['z'][i] = dmrot[int(facepoints_l['y'][i]), int(facepoints_l['x'][i])]

    min_z_l = statistics.median(facepoints_l['z'])-100
    max_z_l = statistics.median(facepoints_l['z'])+100
    mask[XYZ[:,:,2]>max_z_l] = 0
    mask[XYZ[:,:,2]<min_z_l] = 0

    kernel = np.ones((9,9),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 2)
    
    maskoutn, maskout = cv2.connectedComponents(mask)
    max1 = 0
    imax = 0
    for i in range(1, maskoutn):
        if len(maskout[maskout==i])>max1:
            max1 = len(maskout[maskout==i])
            imax = i
    mask[maskout!=imax] = 0
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    return (XYZ, dispmap, mask, fixedLeft, fixedRight, leftMapX, leftMapY)

### Загружаем фотографии и данные о калибровке, опционально делаем цветовую калибровку

calr = np.load ('26.05/calib_lr.npz')
calt = np.load ('26.05/calib_lt.npz')
colcal = np.load('26.05/color_calib.npz')
lFrame = cv2.imread('26.05/for_rectification/left.png')
rFrame = cv2.imread('26.05/for_rectification/right.png')
tFrame = cv2.imread('26.05/for_rectification/top.png')

# rFrame = color_correction(rFrame,colcal['ar'])
# tFrame = color_correction(tFrame,colcal['at'])
# cv2.imwrite('26.05/for_rectification/right_cc.png', rFrame)
# cv2.imwrite('26.05/for_rectification/top_cc.png', tFrame)
# lFramelayer = cv2.cvtColor(lFrame, cv2.COLOR_BGR2GRAY)
# rFramelayer = cv2.cvtColor(np.uint8(rFrame), cv2.COLOR_BGR2GRAY)
# tFramelayer = cv2.cvtColor(np.uint8(tFrame), cv2.COLOR_BGR2GRAY)
# for i in range(0,3):
#     lFrame[:,:,i] = lFramelayer
#     rFrame[:,:,i] = rFramelayer
#     tFrame[:,:,i] = tFramelayer

XYZlr, dispmaplr, masklr, fixedLeftlr, fixedRightlr, leftMapXlr, leftMapYlr = stacker(lFrame, rFrame, calr, False)
XYZlt, dispmaplt, masklt, fixedLeftlt, fixedRightlt, leftMapXlt, leftMapYlt = stacker(lFrame, tFrame, calt, True)

# Генерируем поляризационный инпут

ymax = lFrame.shape[0]
xmax = lFrame.shape[1]
fullmask = np.zeros((ymax, xmax, 2))
pol3d = np.zeros((ymax, xmax, 3))
pol3dunr = np.zeros((ymax, xmax, 3))
dmstack = np.zeros((ymax,xmax,2))
xstack = np.zeros((ymax,xmax,2))
ystack = np.zeros((ymax,xmax,2))

fixedRightlt1 = np.rot90(fixedRightlt)

for y in range (0,ymax):
    for x in range(0,xmax):
        if masklr[y,x] == 1:
            dx = dispmaplr[y,x]
            x1 = int(round(x - dx,0))
            pol3dunr[y,x,1] = fixedRightlr[y,x1,0]
        
        if masklt[y,x] == 1:
            dy = dispmaplt[y,x]
            y1 = int(round(y - dy,0))
            pol3dunr[y,x,2] = fixedRightlt1[y1,x,0]
            
for y in range (0,ymax):
    for x in range(0,xmax):
        ylr = int(round(leftMapYlr[y,x], 0))
        if ((ylr>=ymax) or (ylr<0)):
            continue
            
        xlr = int(round(leftMapXlr[y,x], 0))
        if ((xlr>=xmax) or (xlr<0)):
            continue
            
        ylt = int(round(leftMapYlt[y,x], 0))
        if ((ylt>=ymax) or (ylt<0)):
            continue
            
        xlt = int(round(leftMapXlt[y,x], 0))
        if ((xlt>=xmax) or (xlt<0)):
            continue
            
        if masklr[y,x] == 1:
            fullmask[ylr,xlr,0] = 1
            pol3d[ylr,xlr,1] = pol3dunr[y,x,1]
            xstack[ylr,xlr,0] = XYZlr[y,x,0]
            ystack[ylr,xlr,0] = XYZlr[y,x,1]
            dmstack[ylr,xlr,0] = XYZlr[y,x,2]
            
        if masklt[y,x] == 1:
            fullmask[ylt,xlt,1] = 1
            pol3d[ylt,xlt,2] = pol3dunr[y,x,2]
            xstack[ylt,xlt,1] = XYZlt[y,x,0]
            ystack[ylt,xlt,1] = XYZlt[y,x,1]
            dmstack[ylt,xlt,1] = XYZlt[y,x,2]
        
        pol3d[y,x,0] = lFrame[y,x,0]

newmask = fullmask[:,:,0] + fullmask[:,:,1]

newmask[newmask<2] = 0
newmask[newmask==2] = 1

pol3dfin = np.zeros((ymax, xmax, 3))
pol3dfin[:,:,0] = pol3d[:,:,0]*newmask
pol3dfin[:,:,1] = pol3d[:,:,1]*newmask
pol3dfin[:,:,2] = pol3d[:,:,2]*newmask

xm = (xstack[:,:,0] + xstack[:,:,1])/2
ym = (ystack[:,:,0] + ystack[:,:,1])/2
dm = (dmstack[:,:,0] + dmstack[:,:,1])/2

xm = xm * newmask
ym = ym * newmask
dm = dm * newmask

XYZm = np.zeros(XYZlr.shape)
XYZm[:,:,0] = xm
XYZm[:,:,1] = ym
XYZm[:,:,2] = dm
N = get_kdtree_normals(XYZm, newmask)
azimuth = math.atan2(-N[:,:,1],N[:,:,0])
zenith = math.acos(N[:,:,2])

# Вычисляем поляризационные нормали
phi, theta, ro = utils.polarimgs2norms(pol3dfin, xvals, n, newmask)

# Корректируем поляризационные нормали
phi_corr = fuser(phi, newmask, azimuth)
grad_corr, norms_corr = normals(-phi_corr, theta)

### ИНТЕГРАЦИЯ

# Устанавливаем коэффициенты

clambda = 0.015
calpha = 0.001

surface = fast_spanning_tree_integrator(-norms_corr[:,:,0],-norms_corr[:,:,1],dm,clambda,ro,calpha)



