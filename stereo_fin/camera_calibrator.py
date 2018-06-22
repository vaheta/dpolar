import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import argparse
import json

class StereoCalibration(object):
    def __init__(self, filepath1, filepath2, chess_cell_size = 20):
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objp = self.objp*chess_cell_size

        self.objpoints = []  
        self.imgpoints_l = []  
        self.imgpoints_r = []  
        self.img_shape = 0

        self.cal1_path = filepath1
        self.cal2_path = filepath2
        self.read_images(self.cal1_path, self.cal2_path)

    def read_images(self, cal1_path, cal2_path):
        images_left = glob.glob(cal1_path + '*.png')
        images_right = glob.glob(cal2_path + '*.png')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            print ('Шаг ', i)
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = img_l[:,:,0]
            gray_r = img_r[:,:,0]
            
            fl_l = False
            fl_r = False

            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)

                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)
                cv2.imwrite(('out_l/' + str(i) + '.png'), ret_l)
                fl_l = True

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)

                ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r, ret_r)
                cv2.imwrite(('out_r/' + str(i) + '.png'), ret_r)
                fl_r = True
            
            if fl_l and fl_r:
                self.objpoints.append(self.objp)
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)
            
            img_shape = gray_l.shape[::-1]
        self.img_shape = img_shape
        return (self.objpoints, self.imgpoints_l, self.imgpoints_r, self.img_shape)

def calibrate(dims, objpoints, imgpoints_l, imgpoints_r):
    print (len(objpoints))
    print (len(imgpoints_l))
    print (len(imgpoints_r))
    rt, M1, d1, r1, t1 = cv2.calibrateCamera(objpoints, imgpoints_l, dims, None, None)
    rt, M2, d2, r2, t2 = cv2.calibrateCamera(objpoints, imgpoints_r, dims, None, None)
    
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l,
        imgpoints_r, M1, d1, M2, d2, dims,
        criteria=stereocalib_criteria, flags=flags)
    
    # print('Intrinsic_mtx_1', M1)
    # print('dist_1', d1)
    # print('Intrinsic_mtx_2', M2)
    # print('dist_2', d2)
    # print('R', R)
    # print('T', T)
    # print('E', E)
    # print('F', F)

    # print('')

    camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                        ('dist2', d2), ('rvecs1', r1),
                        ('rvecs2', r2), ('R', R), ('T', T),
                        ('E', E), ('F', F)])

    return camera_model