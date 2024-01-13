# Object Detection models 
# DetectNet-COCO-Airplane    coco-airplane
# SSD-Mobilenet-v2           ssd-mobilenet-v2  ssd-mobilenet-v1
# SSD-Inception-v2           ssd-inception-v2

import jetson.inference
import jetson.utils
import cv2
import numpy as np

# watch https://www.youtube.com/watch?v=mB025B7KpeE&t=1206s to complete the code

net = jetson.inference.detectNet("ssd-inception-v2", threshold = 0.5)
dispW = 1280#640
dispH = 720 # 480
flip = 0
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv2.VideoCapture(camSet)
# cap.set(3,640)
# cap.set(4,480)

while True:
    ret, img = cap.read()
    if ret == False:
        continue
    
    imgCuda = jetson.utils.cudaFromNumpy(img)
    detections = net.Detect(imgCuda, overlay = "OVERLAY_NONE")
    print(len(detections))
    for d in detections:
        x1,y1,x2,y2= int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        #print(d.ClassID,d.Confidence)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.putText(img,className, (x1+5,y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,255),2)
        cv2.putText(img,f'FPS: {int(net.GetNetworkFPS())}', (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
        #print(type(d.Confidence))
    
    
    #img = jetson.utils.cudaToNumpy(imgCuda)

    cv2.imshow("output", img)
    cv2.waitKey(1)
    if cv2.waitKey(40) == 27:
        break
cv2.destroyAllWindows()
cap.release()