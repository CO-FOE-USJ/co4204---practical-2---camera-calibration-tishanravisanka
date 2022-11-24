# K.T.R. Wickramasinghe
# 18/ENG/118

import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
 
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


rows = 6 # chess board rows.
columns = 9 # chess board columns.
world_scaling = 22.1 # square size

# squares coordinates in the chess board world space
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling* objp

width = 2448
height = 2048

def calibrateCamera(images_folder):
    print((images_folder.split('/')[1]).split('*')[0])
    images_names = glob.glob(images_folder)
    images = []
    calibratedImages = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
 
    # Arrays to store object points and image points from all the images.
    imgPoints = [] # 2d points in image plane.
    objPoints = [] # 3d point in real world space

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #opencv can attempt to improve the chess board coordinates
            #Convolution size (11, 11) used to improve corner detection
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)

            calibratedImages.append(frame)
 
            objPoints.append(objp)
            imgPoints.append(corners)
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, (width, height), None, None)
 
    return mtx, dist ,imgPoints, objPoints, calibratedImages
 
def stereoCalibrate(mtx1, dist1, imgpoints_left, mtx2, dist2, imgpoints_right, objpts):
    
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, distC1, CM2, distC2, R, T, E, F = cv.stereoCalibrate(objpts, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (2448,2048), criteria = criteria, flags = stereocalibration_flags)
 

    return CM1, distC1, CM2, distC2, R, T, E, F
 
def rectification(CM1, distC1, CM2, distC2, R, T):
    leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap, leftROI, rightROI = cv.stereoRectify(CM1, distC1, CM2,
                                                                                    distC2, (width, height), R, T,
                                                                                    flags=cv.CALIB_ZERO_DISPARITY,
                                                                                    alpha=0.9)
    
    leftMap1, leftMap2 = cv.initUndistortRectifyMap(CM1, distC1, leftRectification, leftProjection, (width, height), cv.CV_16SC2)
    rightMap1, rightMap2 = cv.initUndistortRectifyMap(CM2, distC2, rightRectification, rightProjection, (width, height), cv.CV_16SC2)

    return leftMap1, leftMap2, rightMap1, rightMap2

def fileWrite(file, title, data):
    file.write(title + "\n\n")
    for i in data:
        file.write(str(i))
        file.write("\n")
    file.write("\n\n")

def saveProperties(CM1, distC1, CM2, distC2, R, T, E, F):

    file=open("Parameters/cameraparameters.txt","w")
    fileWrite(file, "cameraMatrix1 - Left camera intrinsics", CM1)
    fileWrite(file, "distCoeffs1 - Left camera distortion coefficients", distC1)
    fileWrite(file, "cameraMatrix2 - Right camera intrinsics", CM2)
    fileWrite(file, "distCoeffs2 - Right camera distortion coefficients", distC2)
    fileWrite(file, "R - Rotation matrix", R)
    fileWrite(file, "T - Translation vector", T)
    fileWrite(file, "E - Essential Matrix", E)
    fileWrite(file, "F - Fundamental Matrix", F)


def drawLines(leftMap1, leftMap2, rightMap1, rightMap2, calibratedImages1, calibratedImages2):

    for leftImg, rightImg in zip(calibratedImages1, calibratedImages2):

        good_pt_left = cv.remap(leftImg, leftMap1, leftMap2, cv.INTER_LINEAR)
        good_pt_right = cv.remap(rightImg, rightMap1, rightMap2, cv.INTER_LINEAR)

        # image combining
        concat_img = cv.hconcat([good_pt_left, good_pt_right])

        cropped_image = concat_img[300:1800, 600:4300]
        croppedHeight = cropped_image.shape[0]
        croppedWidth = cropped_image.shape[1]

        # drawing lines
        for i in range(0, croppedHeight, 10):
            cv.line(cropped_image, (0, i), (croppedWidth, i), (0, 255, 0))

        cv.destroyAllWindows()
        cv.namedWindow("Rectified Image", cv.WINDOW_NORMAL)
        cv.resizeWindow("Rectified Image", 500, 400)
        cv.imshow('Rectified Image', cropped_image)
        cv.waitKey(8000)
        


mtx1, dist1, imgPoints1, objPoints1, calibratedImages1 = calibrateCamera(images_folder = 'Chessboard/left*.bmp')
mtx2, dist2, imgPoints2, objPoints2, calibratedImages2 = calibrateCamera(images_folder = 'Chessboard/right*.bmp')
 
CM1, distC1, CM2, distC2, R, T, E, F = stereoCalibrate(mtx1, dist1, imgPoints1, mtx2, dist2, imgPoints2, objPoints1)
 
leftMap1, leftMap2, rightMap1, rightMap2= rectification(CM1, distC1, CM2, distC2, R, T)

drawLines(leftMap1, leftMap2, rightMap1, rightMap2, calibratedImages1, calibratedImages2)

saveProperties(CM1, distC1, CM2, distC2, R, T, E, F)