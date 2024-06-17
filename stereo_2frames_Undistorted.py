import argparse
import numpy as np
from cv2 import *
# def Mousecallback(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print (threeD[y][x])

#--------------calibration parameter----------------------
cameraMatrix_Left=np.array(([2.3892563e+002,0,1.5892866e+002],[0,2.3865967e+002,1.3122136e+002],[0,0,1]))
distCoeffs_Left=np.array([-0.43373,0.16967,0.00095,-0.00027,0])
cameraMatrix_Right=np.array(([2.3892563e+002,0,1.6092744e+002],[0,2.3865967e+002,1.2619615e+002],[0,0,1]))
distCoeffs_Right=np.array([-0.43376,0.16801,0.00007,-0.00042,0])

R=np.array([-0.00870,-0.00792,0.00152])
R = cv2.Rodrigues(R)[0]
T=np.array([-0.06184,-0.00025,0.0008])

#--------------necessary parameter----------------------

size=(320,240)
camera0=VideoCapture(1)
camera0.set(3,320)
camera0.set(4,240)
camera1=VideoCapture(2)
camera1.set(3,320)
camera1.set(4,240)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix_Left, distCoeffs_Left, cameraMatrix_Right, distCoeffs_Right, size, R, T)
left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix_Left, distCoeffs_Left, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix_Right, distCoeffs_Right, R2, P2, size, cv2.CV_16SC2)

# cv2.namedWindow("left")
# cv2.namedWindow("right")
# cv2.namedWindow("depth")
# cv2.moveWindow("left", 0, 0)
# cv2.moveWindow("right", 600, 0)
# cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
# cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
#--------------detection loop----------------------
while 1:
    ret0, frame0 = camera0.read()
    ret1, frame1 = camera1.read()
    frame0 = cvtColor(frame0, COLOR_BGR2GRAY)
    frame1 = cvtColor(frame1, COLOR_BGR2GRAY)
    img0_rectified = cv2.remap(frame0, left_map1, left_map2, cv2.INTER_LINEAR)
    img1_rectified = cv2.remap(frame1, right_map1, right_map2, cv2.INTER_LINEAR)
    # num = cv2.getTrackbarPos("num", "depth")
    # blockSize = cv2.getTrackbarPos("blockSize", "depth")
    # if blockSize % 2 == 0:
    #     blockSize += 1
    # if blockSize < 5:
    #     blockSize = 5
    #
    # stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    # disparity = stereo.compute(img0_rectified, img1_rectified)
    #
    # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    # threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
    imshow('left', img0_rectified)
    imshow('right', img1_rectified)
    # imshow('depth',disp)
    # cv2.setMouseCallback("depth", Mousecallback, None)
    if waitKey(30)==27:
        break
