import cv2
import numpy as np
import math
from plyfile import PlyData, PlyElement
import subprocess
import os
import json
import statistics
import configparser
import camera_calibrator as cc
pi = 3.14159265


def findim(fld, cam):
    for filename in os.listdir(fld):
        if cam in filename:
            return filename
        
def xyz2pc (XYZ, im, mask, sfm):
    pcdt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    fa = np.empty(len(mask[mask>0]), dtype=pcdt)
    i = 0
    r = np.reshape(np.array([sfm[0:9]]),(3,3))
    t = np.array([sfm[9:]])
    for x in range(0, XYZ.shape[1]):
        for y in range(0, XYZ.shape[0]):
            if (mask[y,x]==1):
                xyznew = XYZ[y,x,:]*r+t
                fa[i] = np.array([(xyznew[0], xyznew[1], xyznew[2], im[y,x,0], im[y,x,1], im[y,x,2])], dtype=pcdt)
                i = i + 1
    return fa

def call_PSMNet(imgL_path, imgR_path, disp_path, PSMNet_folder_path = '../PSMNet/', maxdisp = 192, imsize = 1024, mirror=0):
    # Вызываем PSMNet 

    command = ['python2', PSMNet_folder_path + 'PSMNet_stereo.py', 
               '--maxdisp', str(maxdisp), 
               '--loadmodel', PSMNet_folder_path + 'models/pretrained_model_KITTI2015.tar', 
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
        # Детектируем корректного человека
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

def stereo_recon(lFrame, rFrame, calp, PSMNet_folder_path, rotate=False, maxdisp=192, size0=1024):
    
    ### Ректификация
    img_shape = lFrame[:,:,0].shape[::-1]
    (RCT1, RCT2, P1, P2, dsp2dm, ROI1, ROI2) = cv2.stereoRectify(
            calp['M1'], calp['dist1'],
            calp['M2'], calp['dist2'],
            img_shape, calp['R'], calp['T'],
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, alpha=-1)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            calp['M1'], calp['dist1'], RCT1,
            P1, img_shape, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            calp['M2'], calp['dist2'], RCT2,
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
        print ("Внимание! Вероятно, неудачное значение maxdisp")

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
    call_PSMNet(imgL_cut_path, imgR_cut_path, disp_path, PSMNet_folder_path=PSMNet_folder_path, imsize=size0, maxdisp=maxdisp)
    dispmap_load = np.load(disp_path)
    
    ### Конвертируем карту смещений в карту глубин

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
    
    return (XYZ, dispmap, mask, fixedLeft)


def main():

    ### Считываем конфигурационный файл
    print (50*'-')
    print ('Считываю конфигурационный файл')
    config = configparser.ConfigParser()
    config.read('config.ini')
    if (config.sections()==[]):
        print ('Ошибка: конфигурационный файл config.ini не найден, или в нем отсутствуют записи.')
        return
    else:
        try:
            size0 = int(config['PSMNet']['size0'])
            maxdisp = int(config['PSMNet']['maxdisp'])
            mirror = int(config['PSMNet']['vertical'])
            PSMNet_folder_path = config['PSMNet']['PSMNet_folder_path']
            images_folder_path = config['PSMNet']['images_folder_path']
        except:
            print ('Ошибка: в конфигурационном файле config.ini отсутствуют необходимые записи раздела PSMNet.')
            return
        
        try:
            output_path = config['3D']['output_path']
        except:
            print ('Ошибка: в конфигурационном файле config.ini отсутствуют необходимые записи раздела 3D.')
            return
    
    ### Производим калибровку системы

    if ('CALIB' in config):
        try:
            calib_folder = config['CALIB']['calib_folder']
            square_size = int(config['CALIB']['square_size_mm'])
        except:
            print ('Ошибка: укажите путь до папки с калибровочными данными в конфигурационном файле config.ini.')
            return
    else:
        print ('Ошибка: укажите путь до папки с калибровочными данными в конфигурационном файле config.ini.')
        return

    try:
        cam_ids = json.loads(config['CALIB']['cam_ids'])
    except:
        print ('Ошибка считывания id камер в конфигурационном файле config.ini.')
        return
    
    if (len(cam_ids)%2!=0):
        print ('Ошибка считывания id камер в конфигурационном файле config.ini.')
        return
    
    sfm = {}
    if len(cam_ids)>2:
        for i in range(0, len(cam_ids)):
            try:
                sfm[cam_ids[i]] = json.loads(config['SfM'][str(cam_ids[i])])
            except:
                print ('Ошибка считывания SfM параметров в конфигурационном файле config.ini.')
                return
    else:
        zerotr = [1,0,0,0,1,0,0,0,1,0,0,0]
        for i in range(0, cam_ids):
            sfm[cam_ids[i]] = zerotr
    print ('Конфигурационный файл успешно считан')
    print (50*'-')

    catbs = []
    for i in range(0, len(cam_ids)//2):
        cam_cal_path = calib_folder + str(cam_ids[i]) + '-' + str(cam_ids[i+1])
        if os.path.isfile(cam_cal_path):
            print ('Считываю калибровочные файлы для камер', cam_ids[i], cam_ids[i+1])
            catb = np.load (cam_cal_path)
            catb['tframe'] = cam_ids[i]
            catb['bframe'] = cam_ids[i+1]
            print ('Калибровка для камер', cam_ids[i], cam_ids[i+1], 'успешно считана')
        else:
            print ('Калибровочный файл для камер', cam_ids[i], cam_ids[i+1], 'не найден. Произвожу калибровку по фото')
            path_t = calib_folder + str(cam_ids[i]) + '/'
            path_b = calib_folder + str(cam_ids[i+1]) + '/'
            print (path_t, path_b)
            caltemp = cc.StereoCalibration(path_t, path_b, square_size)
            catb = cc.calibrate(caltemp.img_shape, caltemp.objpoints, caltemp.imgpoints_l, caltemp.imgpoints_r)
            np.savez(cam_cal_path, 
                M1 = catb['M1'], M2 = catb['M2'],
                d1 = catb['dist1'], d2 = catb['dist2'],
                r1 = catb['rvecs1'], r2 = catb['rvecs2'],
                R = catb['R'], T = catb['T'],
                E = catb['E'], F = catb['F'])
            catb['tframe'] = cam_ids[i]
            catb['bframe'] = cam_ids[i+1]
            print ('Калибровка для камер', cam_ids[i], cam_ids[i+1], 'успешно произведена')

        catbs.append(catb)

        print ('Произвожу стереореконструкцию для камер', cam_ids[i], cam_ids[i+1])
        tFrame = cv2.imread(findim(images_folder_path, cam_ids[i]))
        bFrame = cv2.imread(findim(images_folder_path, cam_ids[i]))
        XYZtb, dispmaptb, masktb, fixedLefttb = stereo_recon(tFrame, bFrame, catb, PSMNet_folder_path, mirror, maxdisp, size0)
        
        ### Генерируем облако точек
        print ('Генерирую облако точек для камер', cam_ids[i], cam_ids[i+1])
        fatemp = xyz2pc(XYZtb, fixedLefttb, masktb, sfm[cam_ids[i]])
        if i==0:
            fa = fatemp
        else:
            fa = np.vstack([fa,fatemp])

    ### Сохраняем облако точек
    el = PlyElement.describe(fa, 'vertex')
    PlyData([el], text=True).write(output_path)


if __name__ == '__main__':
   main()