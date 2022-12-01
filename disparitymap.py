# K.T.R. Wickramasinghe
# 18/ENG/118

import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from tqdm import *

width = 2448
height = 2048

BLOCK_SIZE = 100
SEARCH_BLOCK_SIZE = 56


def noramlize_cross_correlation(pixel1, pixel2):
    if pixel1.shape != pixel2.shape:
        return -1
    return (np.sum(pixel1*pixel2)/math.sqrt(np.sum(pow(pixel1,2))*np.sum(pow(pixel2,2))))


# return the index of the block where ncc is max by comparing the blocks
def nccAlgo(y, x, block_left, right_array, block_size=5):
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    first = True
    max_ncc = None
    max_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size, x: x+block_size]
        ncc = noramlize_cross_correlation(block_left, block_right)
        if first:
            max_ncc = ncc
            max_index = (y, x)
            first = False
        else:
            if ncc > max_ncc:
                max_ncc = ncc
                max_index = (y, x)
    return max_index

# geting the sum of absolute pixel block difference
def pixelBlockSumDiff(pixel1, pixel2):
    if pixel1.shape != pixel2.shape:
        return -1
    return np.sum(abs(pixel1 - pixel2))


def sadAlgo(y, x, block_left, right_array, block_size=5):
    # right image search range
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        sad = pixelBlockSumDiff(block_left, block_right)
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)
    return min_index

def disparityMapCal(left_array, right_array):
    width =612
    height = 512
    disparity_map = np.zeros((width, height))
    for y in tqdm(range(BLOCK_SIZE, width-BLOCK_SIZE)):
        for x in range(BLOCK_SIZE, height-BLOCK_SIZE):
            block_left = left_array[y:y + BLOCK_SIZE,
                                    x:x + BLOCK_SIZE]
            min_index = sadAlgo(y, x, block_left,
                                       right_array,
                                       block_size=BLOCK_SIZE)

            disparity_map[y, x] = abs(min_index[1] - x)

    cv.imwrite("disparity.bmp",disparity_map)
    cv.destroyAllWindows()
    cv.namedWindow("Disparity Map", cv.WINDOW_NORMAL)
    cv.resizeWindow("Disparity Map", 1000, 1000)
    cv.imshow("Disparity Map",disparity_map)
    cv.waitKey(0)


def loadCameraParameters():
    inputs=[]
    input_file = open("Parameters/cameraparameters.txt","r")
    for line in input_file:
        inputs.append(line)
    
    CM1= []
    distC1= []
    CM2= []
    distC2= []
    R= []
    T= []

    i=0
    while i<len(inputs):
        if inputs[i].rstrip('\n')=="cameraMatrix1 - Left camera intrinsics":
            i+=2
            while(inputs[i] != '\n'):
                CM1.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
 
        elif inputs[i].rstrip('\n')=="distCoeffs1 - Left camera distortion coefficients":
            i+=2
            while(inputs[i] != '\n'):
                distC1.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
        elif inputs[i].rstrip('\n')=="cameraMatrix2 - Right camera intrinsics":
            i+=2
            while(inputs[i] != '\n'):
                CM2.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
        elif inputs[i].rstrip('\n')=="distCoeffs2 - Right camera distortion coefficients":
            i+=2
            while(inputs[i] != '\n'):
                distC2.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
        elif inputs[i].rstrip('\n')=="R - Rotation matrix":
            i+=2
            while(inputs[i] != '\n'):
                R.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
        elif inputs[i].rstrip('\n')=="T - Translation vector":
            i+=2
            while(inputs[i] != '\n'):
                T.append(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", inputs[i]))))
                i+=1
        else:
            i+=1

    return np.array(CM1), np.array(distC1), np.array(CM2), np.array(distC2), np.array(R), np.array(T)

def rectification(CM1, distC1, CM2, distC2, R, T):
    leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap, leftROI, rightROI = cv.stereoRectify(CM1, distC1, CM2,
                                                                                    distC2, (width, height), R, T,
                                                                                    flags=cv.CALIB_ZERO_DISPARITY,
                                                                                    alpha=0.9)
    # undistortion
    leftMap1, leftMap2 = cv.initUndistortRectifyMap(CM1, distC1, leftRectification, leftProjection, (width, height), cv.CV_16SC2)
    rightMap1, rightMap2 = cv.initUndistortRectifyMap(CM2, distC2, rightRectification, rightProjection, (width, height), cv.CV_16SC2)

    return leftMap1, leftMap2, rightMap1, rightMap2

def read_images(images_path):
    images_names = glob.glob(images_path)
    images = []
    for imname in images_names:
        print()
        im = cv.imread(imname, 1)
        # brightnessincrease
        im = cv.normalize(im,None,alpha=1.0,beta=200, norm_type=cv.NORM_MINMAX)

         #convert to gray
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        images.append(gray)

    return images


def remaping(leftMap1, leftMap2, rightMap1, rightMap2, calibratedImages1, calibratedImages2):

    rectified_img_left = [] 
    rectified_img_right = []
    for leftImg, rightImg in zip(calibratedImages1, calibratedImages2):

        # remaping
        good_pt_left = cv.remap(leftImg, leftMap1, leftMap2, cv.INTER_LINEAR)
        good_pt_right = cv.remap(rightImg, rightMap1, rightMap2, cv.INTER_LINEAR)

        # resize image
        good_pt_left = cv.resize(good_pt_left, (612,512), interpolation = cv.INTER_AREA)
        good_pt_right = cv.resize(good_pt_right, (612,512), interpolation = cv.INTER_AREA)

        rectified_img_left.append(good_pt_left) 
        rectified_img_right.append(good_pt_right)

    return rectified_img_left, rectified_img_right


def disparityMap(leftImgs, rightImgs):

    for leftImg, rightImg in zip(leftImgs, rightImgs):
        
        # ncc or sad
        # disparityMapCal(leftImg,rightImg)


        window_size = 5
        stereo = cv.StereoSGBM_create(
            minDisparity=-16,
            numDisparities=80,
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=10,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )


        disparity_map = stereo.compute(leftImg,rightImg).astype(np.float32)/16.0
        cv.imwrite("SGBM.bmp",disparity_map)
        cv.namedWindow("Disparity Map", cv.WINDOW_NORMAL)
        cv.resizeWindow("Disparity Map", 1000, 1000)
        cv.imshow("Disparity Map",disparity_map)
        cv.waitKey(0)





img_left = read_images("SceneImages/left*.bmp")
img_right = read_images("SceneImages/right*.bmp")

CM1, distC1, CM2, distC2, R, T = loadCameraParameters()

leftMap1, leftMap2, rightMap1, rightMap2= rectification(CM1, distC1, CM2, distC2, R, T)

rectified_img_left, rectified_img_right = remaping(leftMap1, leftMap2, rightMap1, rightMap2, img_left, img_right)

disparityMap(rectified_img_left, rectified_img_right)



