import cv2
import numpy as np
import math
import utils
from plyfile import PlyData, PlyElement
import subprocess
import json
import os
import shutil
pi = 3.14159265

def run_openpose(imdir, outdir):
    command = ['/home/vaheta/builds/openpose/build/examples/openpose/openpose.bin', 
           '--image_dir', imdir, 
           '--face', 
           '--net_resolution', '656x368', 
           '--write_json', outdir,
           '--display', '0']
    subprocess.call(command)
    
def read_keypoint_jsons(jsondir):
    faces = {}
    for filename in os.listdir(jsondir):
        if filename.endswith(".json"):
            with open((jsondir+'/'+filename)) as f:
                data = json.load(f)
                tempdict = {'x':[], 'y':[], 'c':[]}
                for i,k in enumerate(data['people'][0]['face_keypoints_2d']):
                    if i%3==0:
                        tempdict['x'].append(k)
                    elif i%3==1:
                        tempdict['y'].append(k)
                    elif i%3==2:
                        tempdict['c'].append(k)
                faces[filename[0:-15]] = tempdict
    return faces

# Set paths to stereo files here:

imgL_path = 'face_render/facex005.png'
imgR_path = 'face_render/face0.png'

procdir = '/home/vaheta/dpolar/for_processing'


if os.path.isdir(procdir):
    shutil.rmtree(procdir)
os.makedirs(procdir)

shutil.copyfile(imgL_path, (procdir+'/imgl.png'))
shutil.copyfile(imgR_path, (procdir+'/imgr.png'))

run_openpose(procdir,procdir)