from cv2 import *
from threading import Thread, currentThread, activeCount
import numpy as np
webcam_address0="rtsp://192.168.1.105:6554/stream_0"
webcam_address1="rtsp://192.168.1.106:6554/stream_0"
cap0=VideoCapture(webcam_address0)
cap1=VideoCapture(webcam_address1)
count=0
flag_l=False
flag_r=False
flag_stop=1
def webcam0_read():
    global frame0
    global flag_l
    global flag_stop
    while flag_stop==1:
        print('start reading l')
        ret0, frame0 = cap0.read()
        print('l done')
        flag_l=True
def webcam1_read():
    global frame1
    global flag_r
    global flag_stop
    while flag_stop==1:
        print('start reading r')
        ret1, frame1 = cap1.read()
        print('r done')
        flag_r=True
t0=Thread(target=webcam0_read)
t0.setDaemon(True)
t0.start()
t1=Thread(target=webcam1_read)
t1.setDaemon(True)
t1.start()
while 1:
    print('ready to print!')
    if flag_l==True and flag_r==True:
        frame0_r=resize(frame0,(640,360))
        frame1_r=resize(frame1,(640,360))
        imshow("Video Stream0", frame0_r)
        imshow("Video Stream1", frame1_r)
        if waitKey(5) == 53 or waitKey(5) == 73:
            path0=str(count)+"l.jpg"
            path1=str(count)+"r.jpg"
            imwrite(path0,frame0)
            imwrite(path1,frame1)
            count+=1
        elif waitKey(5)==27:
            break
flag_stop=0