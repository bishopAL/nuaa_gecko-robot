from cv2 import *
from threading import Thread
import numpy as np
#------------------------important parameter define-----------
startX = 0.0
startY = 0.0
endX = 0.0
endY = 0.0
flag_detection=False
def detection(img):
    global startX
    global startY
    global endX
    global endY
    global flag_detection
    flag_detection=False
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx]=='bottle':
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                flag_detection=True
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return (startX+endX)/2,(endY+startY)/2,flag_detection
#--------------Load the MobileNetSSD model ----------------------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
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
flag_detection_left=False
flag_detection_right=False
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
        # frame0_g = cvtColor(frame0, COLOR_BGR2GRAY)
        # frame1_g = cvtColor(frame1, COLOR_BGR2GRAY)
        frame0_g=frame0
        frame1_g=frame1
        frame0_rectified = cv2.remap(frame0_g, left_map1, left_map2, cv2.INTER_LINEAR)
        frame1_rectified = cv2.remap(frame1_g, right_map1, right_map2, cv2.INTER_LINEAR)
        frame0_rectified=resize(frame0_rectified,(640,360))
        frame1_rectified=resize(frame1_rectified,(640,360))
        center0_x, center0_y, flag_detection_left = detection(frame0_rectified)
        center1_x, center1_y, flag_detection_right = detection(frame1_rectified)
        if flag_detection_left == True and flag_detection_right == True:
            disparity = center0_x - center1_x
            w = T[0] / disparity
            object_Z = cameraMatrix_Left[0, 0] * w
            object_X = (center0_x - 160) * w
            object_Y = (120 - center0_y) * w
            print("the object is at %f,%f,%f." % (object_X, object_Y, object_Z))
        else:
            print('Cannot detect the bottle.')
        imshow('left', frame0_rectified)
        imshow('right', frame1_rectified)
        if waitKey(30) == 27:
            break
flag_stop=0