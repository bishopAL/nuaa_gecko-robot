import numpy as np
from cv2 import *
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
flag_detection_left=False
flag_detection_right=False
#--------------detection loop----------------------
while 1:
    ret0, frame0 = camera0.read()
    ret1, frame1 = camera1.read()
    #frame0 = cvtColor(frame0, COLOR_BGR2GRAY)
    #frame1 = cvtColor(frame1, COLOR_BGR2GRAY)
    img0_rectified = cv2.remap(frame0, left_map1, left_map2, cv2.INTER_LINEAR)
    img1_rectified = cv2.remap(frame1, right_map1, right_map2, cv2.INTER_LINEAR)
    #print(img1_rectified.shape)
    center0_x, center0_y, flag_detection_left=detection(img0_rectified)
    center1_x, center1_y, flag_detection_right=detection(img1_rectified)
    if flag_detection_left==True and flag_detection_right==True:
        disparity = center0_x - center1_x
        w = T[0] / disparity
        object_Z = cameraMatrix_Left[0,0] * w
        object_X = (center0_x - 160) * w
        object_Y = (120 - center0_y) * w
        print("the object is at %f,%f,%f." % (object_X,object_Y,object_Z))
    else:
        print('Cannot detect the bottle.' )
    imshow('left', img0_rectified)
    imshow('right', img1_rectified)

    if waitKey(30)==27:
        break
