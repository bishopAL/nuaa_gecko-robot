from cv2 import *
from threading import Thread
import numpy as np
webcam_address0="rtsp://192.168.1.103:6554/stream_0"
webcam_address1="rtsp://192.168.1.100:6554/stream_0"
cap0=VideoCapture(webcam_address0)
cap1=VideoCapture(webcam_address1)
count=0
flag_l=False
flag_r=False
flag_stop=1
cameraMatrix_Left=np.array(([721.58384,0,608.73662],[0,719.83341,318.73499],[0,0,1]))
distCoeffs_Left=np.array([-0.40606,0.16736,0.00153,-0.00008,0.00000])
cameraMatrix_Right=np.array(([718.60199,0,623.51802],[0,717.55333,358.67959],[0,0,1]))
distCoeffs_Right=np.array([-0.39532,0.14552,0.00085,0.00084,0.00000])
size=(1280,720)
R=np.array([0.00437,0.00436,-0.03726])
R = cv2.Rodrigues(R)[0]
T=np.array([-0.07125,-0.00022,0.00198])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix_Left, distCoeffs_Left, cameraMatrix_Right, distCoeffs_Right, size, R, T)
left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix_Left, distCoeffs_Left, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix_Right, distCoeffs_Right, R2, P2, size, cv2.CV_16SC2)
def webcam0_read():
    global frame0
    global flag_l
    global flag_stop
    while flag_stop==1:
        ret0, frame0 = cap0.read()
        flag_l=True
def webcam1_read():
    global frame1
    global flag_r
    global flag_stop
    while flag_stop==1:
        ret1, frame1 = cap1.read()
        flag_r=True
t0=Thread(target=webcam0_read)
t0.setDaemon(True)
t0.start()
t1=Thread(target=webcam1_read)
t1.setDaemon(True)
t1.start()
while 1:
    if flag_l==True and flag_r==True:
        frame0_g = cvtColor(frame0, COLOR_BGR2GRAY)
        frame1_g = cvtColor(frame1, COLOR_BGR2GRAY)
        frame0_rectified = cv2.remap(frame0_g, left_map1, left_map2, cv2.INTER_LINEAR)
        frame1_rectified = cv2.remap(frame1_g, right_map1, right_map2, cv2.INTER_LINEAR)
        frame0_r=resize(frame0_rectified,(640,360))
        frame1_r=resize(frame1_rectified,(640,360))
        imshow('left', frame0_rectified)
        imshow('right', frame1_rectified)
        if waitKey(30) == 27:
            break
flag_stop=0