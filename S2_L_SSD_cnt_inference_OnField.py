# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
# python 3.7.9 on aircraftenv1

import time
# diff(frame1,frame2): 0.004 sec, createBackgroundSubtractorMOG2: 0.015 sec, objectdetctionYolov4: 0.28 sec
import cv2
import numpy as np
import jetson.inference
import jetson.utils
import os
import datetime
# import matplotlib.pyplot as plt

# from aircraft_detection import model # model = ResNet50(weights='imagenet') you could use better (other) image classifier as well
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
############################################ SSD NMS ###################################################################
# NOW: tracks initiate only when one aircraft is detected (what if at some operations at each frame, several aircraft are
# detected??? YOU SHOULD USE          SSD NMS
################################################  THRESHOLDS  ##########################################################
## CHECK BackgroundSubtractorMOG2(varThreshold = 50) && ALLOWABLE BOX SIZES
## SET EASIER MOTION DETECTION CRITERIA (CURRENT: cv2.contourArea(contour) < 8000 or cv2.contourArea(contour) > 50000 or w/h>5 or w/h <0.5)
## CHECK CONTOUR BOX SIZES FOR DIFFERENT VIDEO SIZES (CAMCORDER IS DIFFERENT FROM GOPRO)
## CHECK SSD/YOLO MODEL
OD_CONFIDENCE_THRESHOLD = 0.7
OD_NMS_THRESHOLD = 0.4 # 0.4
## CHECK SSD/YOLO INPUT SIZE ------------------------
AIRPLANE_DETECTION_CONF_BEFORE_TRACKING = 0.7
## CHECK TRACKING MODEL -----------------------------
AIRPLANE_DETECTION_CONF_AFTER_TRACKING = 0.6
## TRY DIFFERENT OCR MODELS (RECOGNITION MODELS) ----
OCR_CONFIDENCE_THRESHOLD = 0.5
## CHECK DATABASE FOR OTHER COUNTIES (NOT INCLUDED)
## CHECK TRAJECTORY POST PROCESSING && ANYWHERE IN THE CODES FOR: "ONLY FOR SYS 2" UNJAYI K LENGTH ESHO CHECK ...
# MIKONE TA AC INFO RO DISAPPEAR BOKONE BAD AZ YE TIMI (10 FRAME AFTER NEXT AC DETECTION)
AreaLandings = 50000
#MOSSE or CSRT
##################################### image classification #############################################################
#img_path = 'aircraft.jpeg'
#img = image.load_img(img_path, target_size=(224, 224))
##img = cv2.resize(box, (224, 224))
#xx = image.img_to_array(img)
#xx = np.expand_dims(xx, axis=0)
#xx = preprocess_input(xx)
#preds = model.predict(xx)
#print('Predicted:', decode_predictions(preds, top=3)[0])
##################################### motion detection #################################################################
#cap = cv2.VideoCapture('D:\Aircraft detection project\Data Collection/12-16-2020 skypark Sys 1 and Sys 2\GoPro left/all12162020.mp4') #1080 2x 160ft from runway/GH013776.mp4 #'D:/Aircraft detection project/Data Collection/10-30-2020 skypark/1080 2x 160ft from runway/GH013776.mp4'
#cap = cv2.VideoCapture('D:\Aircraft detection project\Data Collection/2021/03-05-2021 skypark sys2 arrival in sky\Gopro sys 2/left-for-presentation.mp4')
print("------------------------------------------1-------------------------------------------------")
dispW = 1280#640
dispH = 720 # 480
flip = 0
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv2.VideoCapture(camSet)
#cap = cv2.VideoCapture('S2 Left/GH014190_Trim2.mp4')

# dispW = 1920#1280#640
# dispH = 1080#720 # 480
# flip = 0
# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cap = cv2.VideoCapture(camSet)

frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 0< x < frame_width (1920) HORIZONTAL AXIS
frame_height =int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 0< y < frame_height (1080) VERTICAL AXIS
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold = 50) # cv2.bgsegm.createBackgroundSubtractorMOG()   varThreshold = between 50 to 100
# check the arguments____________________________________________________________________________________________________
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#out = cv2.VideoWriter("D:\Aircraft detection project\Data Collection/12-16-2020 skypark Sys 1 and Sys 2\GoPro left/allOUT12162020.mp4", fourcc, 20.0, (1920,1080))
out = cv2.VideoWriter("S2 Left/test3.mp4", fourcc, 7.0, (dispW,dispH))
##################################### object detection #################################################################
ODclass_names = []
ODclassFile = 'Models/coco.names'
airplane_class_IDnum = 5
with open(ODclassFile, 'rt') as f:
    ODclass_names = f.read().rstrip('\n').split('\n')
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
configPath = 'Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # try different models when you do that the weights and net.setInput should be adjusted
weightsPath = 'Models/frozen_inference_graph.pb'  # weights
# pip install opencv-python #version 4.4.0.46
# ODnet = cv2.dnn_DetectionModel(weightsPath, configPath)
# ODnet.setInputSize(50, 50)      ##############320 320 or 160 160 or 416 416
# ODnet.setInputScale(1.0 / 127.5)
# ODnet.setInputMean((127.5, 127.5, 127.5))
# ODnet.setInputSwapRB(True)
ODnet = jetson.inference.detectNet("ssd-inception-v2", threshold = 0.5)

# learned from: https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
#OD_CONFIDENCE_THRESHOLD = 0.6
#OD_NMS_THRESHOLD = 0.4 # 0.4
#COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
#ODclass_names= []
#ODclassFile = 'coco.names'
#with open(ODclassFile,'rt') as f:
#    ODclass_names = f.read().rstrip('\n').split('\n')
#airplane_class_IDnum = 4
# pip install opencv-python #version 4.4.0.46
#ODnet = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg") # https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo   # yolo: https://github.com/kiyoshiiriemon/yolov4_darknet
#ODnet = cv2.dnn.readNet("Models/YOLOv4-tiny/yolov4-tiny.weights", "Models/YOLOv4-tiny/yolov4-tiny.cfg")
#ODnet = cv2.dnn.readNet("Models/YOLOv3-tiny/yolov3-tiny.weights", "Models/YOLOv3-tiny/yolov3-tiny.cfg")                   # a little better than tiny yolo
#ODnet = cv2.dnn.readNet("Models/Tiny YOLO/yolov2-tiny.weights", "Models/Tiny YOLO/yolov2-tiny.cfg")
#ODnet = cv2.dnn.readNet("Models/YOLOv3-320/yolov3.weights", "Models/YOLOv3-320/yolov3.cfg")
#ODnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #try other commands
#ODnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#ODnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16) #try other commands
#ODnet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
#ODnet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
#ODnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#ODmodel = cv2.dnn_DetectionModel(ODnet)
#ODmodel.setInputParams(size=(416, 416), scale=1/255)
#ODmodel.setInputParams(size=(608, 608), scale=1/255)
###################################### tracker class ###################################################################
# pip install opencv-contrib-python (for trackers)
#tracker = cv2.TrackerMOSSE_create() #faster   #Tracker Description Blog: https://ehsangazar.com/object-tracking-with-opencv-fd18ccdd7369
#tracker = cv2.TrackerCSRT_create()  # more accurate
# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE'] all image processing based and not deep learning based
# __MEDIANFLOW Tracker: Pros : Excellent tracking failure reporting. Works very well when the motion is predictable and there is no occlusion. Cons : Fails under large motion.
########################################################################################################################




f = -1
i = -1
ii = [0,0,0] # 3 is enough; 5 also could be used
Trajectory = np.array([[0,0]])
T = []
L = 0        #Landings
D = 0        #Departures (take-offs)
Text1 = 'Status : No Operation is Going On'
Text2 = 'No Previous Operation'
Text3 = '# Departures : ' + str(D)
Text4 = '# Landings   : ' + str(L)
aircraftINFO = ''




########################################################################################################################
################################################### OCR ################################################################
########################################################################################################################
# def decodeText2(scores):
#     text = ""
#     alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"
#     for i in range(scores.shape[0]):
#         c = np.argmax(scores[i][0])
#         if c != 0:
#             text += alphabet[c - 1]
#         else:
#             text += '-'

#     # adjacent same letters as well as background text must be removed to get the final output
#     char_list = []
#     charScore_list = []
#     for i in range(len(text)):
#         if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
#             char_list.append(text[i])
#             charScore_list.append(scores[i])
#     return ''.join(char_list), charScore_list

# def decodeText(scores):
#     text = ""
#     alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"
#     for i in range(scores.shape[0]):
#         c = np.argmax(scores[i][0])
#         if c != 0:
#             text += alphabet[c - 1]
#         else:
#             text += '-'

#     # adjacent same letters as well as background text must be removed to get the final output
#     char_list = []
#     for i in range(len(text)):
#         if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
#             char_list.append(text[i])
#     return ''.join(char_list)

# def fourPointsTransform(frame, vertices):
#     vertices = np.asarray(vertices)
#     outputSize = (100, 32)
#     targetVertices = np.array([
#         [0, outputSize[1] - 1],
#         [0, 0],
#         [outputSize[0] - 1, 0],
#         [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

#     rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
#     result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
#     return result

# modelArchFilename = 'Models/textbox.prototxt'
# modelWeightsFilename = 'Models/TextBoxes_icdar13.caffemodel'
# TextRegionDetectorCNN	=	cv2.text.TextDetectorCNN_create(	modelArchFilename, modelWeightsFilename	)

# OCRmodelRecognition = "Models/CRNN_VGG_BiLSTM_CTC.onnx"
# OCRrecognizer = cv2.dnn.readNetFromONNX(OCRmodelRecognition)

########################################################################################################################
###################################### joint analysis 1 (maximum likelihood) ############################################
########################################################################################################################

# from collections import OrderedDict
# import pandas as pd
# #
# database = pd.read_excel ('Models/Database.xls', sheet_name='allcountiesaddedhere') ######################################################################### Do sth and include all counties
# for irn in range(len(database)): # we need this bcz pure number tail numbers (like N65474 or 65474) are imported as number(float) and not string
#     database['n_number'][irn] = str(database['n_number'][irn])
# TailNumberLists = []

# def finalized_tail_number(TNlists):
#     sorted_TNlists = sorted(TNlists, key=TNlists.count, reverse=True)  # sort by frequency
#     shortlist = list(OrderedDict.fromkeys(sorted_TNlists))  # duplicates are removed
#     aircraft_identified = False
#     for tln in shortlist:
#         if len(tln)>1:
#             if database[database['n_number'] == tln[1:]].size == 0: # this tln (tail number) is not in the database
#                 continue
#             else: # this tln will be selected as the finalized tail number
#                 index = database[database['n_number'] == tln[1:]].index.values[0]
#                 aircraft_detailed_info = database.loc[index]
#                 finalized_TN = tln
#                 aircraft_identified = True
#                 break
#     if aircraft_identified == False and len(shortlist) != 0:
#         aircraft_detailed_info = 'Not in Utah Database'
#         finalized_TN = shortlist[0]
#     if aircraft_identified == False and len(shortlist) == 0:
#         aircraft_detailed_info = ''
#         finalized_TN = 'Unrecognized Aircraft'
#     return finalized_TN, aircraft_detailed_info

########################################################################################################################
#########################################        joint analysis 2       ###############################################
########################################################################################################################
# def softmax(vector):
# 	e = np.exp(vector)
# 	return e / e.sum()

# Alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"

# def jointProbabilityDecoder(Det_tailnumber, charScore_list): #Det_tailnumber must be in lowercase letters like 'n120bf'
#     if len(Det_tailnumber) < 2 or len(Det_tailnumber) > 8:
#         most_probable_TN = ''
#         aircraft_detailed_info = 'Error'
#     else:
#         if len(Det_tailnumber) == 8: # makes it 6
#             Det_tailnumber = Det_tailnumber[1:7]     #drop the first and the last character
#             charScore_list = charScore_list[1:7]
#         if len(Det_tailnumber) == 7: # makes it 6
#             if Det_tailnumber[0] == 'n':
#                 Det_tailnumber = Det_tailnumber[0:6] #drop the last character
#                 charScore_list = charScore_list[0:6]
#             else:
#                 Det_tailnumber = Det_tailnumber[1:7] #drop the first character
#                 charScore_list = charScore_list[1:7]

#         Det_tailnumber = Det_tailnumber[1:]          # DROP 'n' since the databse n_numbers do not have 'N'
#         charScore_list = charScore_list[1:]
#         charSoftProb = []
#         for i in range(len(Det_tailnumber)):
#             charSoftProb.append(softmax(charScore_list[i]))

#         probabilities = np.zeros(len(database['n_number']))
#         for index, N_number in enumerate(database['n_number']):
#             if len(Det_tailnumber) == len(N_number):
#                 probability = 0
#                 for ind, char in enumerate(N_number): # N_number is in uppercase
#                     if Det_tailnumber[ind] == char.lower():
#                         #probability += 1 ######### WHY??? why dont we use the actual probability######################################################
#                         c = Alphabet.find(char.lower())
#                         probability = probability + charSoftProb[ind][0][c + 1]
#                     else:
#                         c = Alphabet.find(char.lower())
#                         probability = probability + charSoftProb[ind][0][c+1]

#                 probabilities[index]=probability/len(N_number)

#         if sum(probabilities) == 0: # no N_number found with the same length
#             most_probable_TN = ''
#             aircraft_detailed_info = 'Most Likely Not in Utah Database'
#         else:
#             most_probable_TN_index = np.argmax(probabilities)
#             aircraft_detailed_info = database.loc[most_probable_TN_index]
#             most_probable_TN = 'N' + database['n_number'][most_probable_TN_index]

#     return most_probable_TN, aircraft_detailed_info

# def finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists):
#     most_probable_TN_List = []
#     aircraft_detailed_info_List = []
#     for i in range(len(TailNumberLists)):
#         most_probable_TN, aircraft_detailed_info = jointProbabilityDecoder(TailNumberLists[i], TailNumberCharScoreLists[i])
#         if most_probable_TN != '':
#             most_probable_TN_List.append(most_probable_TN)
#             aircraft_detailed_info_List.append(aircraft_detailed_info)
#     if len(most_probable_TN_List) == 0:
#         finalized_TN = ''
#         aircraft_detailed_infoo = 'Most Likely Not in Utah Database'
#     else:
#         sorted_TNlists = sorted(most_probable_TN_List, key=most_probable_TN_List.count, reverse=True)
#         finalized_TN = sorted_TNlists[0]
#         index = database[database['n_number'] == finalized_TN[1:]].index.values[0]
#         aircraft_detailed_infoo = database.loc[index]
#         #aircraft_detailed_infoo = aircraft_detailed_info_List[0]
#     return finalized_TN, aircraft_detailed_infoo
########################################################################################################################
########################################################################################################################
########################################################################################################################




def save_operation_files(text, BOX_AIRCRAFT):
    Folder = "S2 Left/"+str(datetime.datetime.now())
    os.mkdir(Folder)
    if len(BOX_AIRCRAFT)<6:
        for i in range(len(BOX_AIRCRAFT)):
            cv2.imwrite(Folder+"/"+str(i)+"-"+text+".jpg",BOX_AIRCRAFT[i])
    else:
        for i in range(5):
            step = len(BOX_AIRCRAFT)/5
            cv2.imwrite(Folder+"/"+str(i)+"-"+text+".jpg",BOX_AIRCRAFT[int(i*step)])


aircraftINFO = ''
database = ''


def put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out):
    cv2.putText(frameRD, Text1, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
    cv2.putText(frameRD, Text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
    cv2.putText(frameRD, Text3, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
    cv2.putText(frameRD, Text4, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
    if len(aircraftINFO) == 0:
        pass
    else:
        cv2.putText(frameRD, 'Aircraft Info:', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 250), 4)
        if len(aircraftINFO) == 20:  # aircraftINFO = aircraft_detailed_info = 'Not in Utah Database'
            cv2.putText(frameRD, 'Not in Utah Database', (80, 300 + 35 * (1)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 250, 250), 3)
        if len(aircraftINFO) == 32:  # aircraftINFO = aircraft_detailed_info = 'Most Likely Not in Utah Database'
            cv2.putText(frameRD, 'Most Likely Not in Utah Database', (80, 300 + 35 * (1)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 250, 250), 3)
        else:
            for column_id, elem in enumerate(aircraftINFO):
                cv2.putText(frameRD, database.columns[column_id]+ ': '+str(elem), (80, 300+35*(1+column_id)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 250), 3)
    image = cv2.resize(frameRD, (1920, 1080))
    out.write(image)
    frames = cv2.resize(frameRD, (960, 540))  # Resize image
    cv2.imshow("output", frames)
    cv2.waitKey(1)

Box=[]
cut_in_traffic = False


while True:#length:

    f += 1
    ret, frame = cap.read()
    if ret==False:
       continue
    frameRD = np.array(frame) # frame for RECORD

    ####################################################################################################################
    ############################################# Motion Detection #####################################################
    ####################################################################################################################

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)  # threshold value = 20
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_, contours, _

    bboxs = np.array([[0,0]])
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 8000 or cv2.contourArea(contour) > 500000 or w/h>5 or w/h <0.5: # SET these to an easier condition_____________________________
            continue
        box = frame[ y:y + h, x:x + w,:] # MAKE IT SQUARE__for predication_____________________________
        Box.append(box)
        bboxs = np.append(bboxs, np.array([[x, y]]), axis=0)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

    ii[0:2] = ii[1:3]
    if np.size(bboxs) > 2:  # means at least a box is detected (MOTION DETECTED)
        ii[2] = 1
        Text1 = 'Status : Possible New Operation'
        i += 1

        ####################################################################################################################
        ############################################# Object Detection #####################################################
        ####################################################################################################################
        start = time.time()
        #ODclasses, ODconfs, ODboxes = ODnet.detect(frame, confThreshold=OD_CONFIDENCE_THRESHOLD, nmsThreshold=OD_NMS_THRESHOLD)
        imgCuda = jetson.utils.cudaFromNumpy(frame)
        detections = ODnet.Detect(imgCuda, overlay = "OVERLAY_NONE")
        # ODboxes = list(ODboxes)
        # ODconfs = list(np.array(ODconfs).reshape(1, -1)[0])
        # ODconfs = list(map(float, ODconfs))
        # indices = cv2.dnn.NMSBoxes(ODboxes, ODconfs, OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD)
        end = time.time()
        print('SSD')
        print(end - start)
        if len(detections) != 0:
            one_airplane_detected=0
            airplane_scores = 0
            for d in detections:
                
                x1,y1,x2,y2= int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)
                className = ODnet.GetClassDesc(d.ClassID)
                color = COLORS[int(d.ClassID) % len(COLORS)]
                #print(d.ClassID,d.Confidence)
                cv2.rectangle(frameRD,(x1,y1),(x2,y2),(255,0,255),2)
                cv2.putText(frameRD,className, (x1,y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.75, color,2)
                cv2.putText(frameRD,f'FPS: {int(ODnet.GetNetworkFPS())}', (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)

                if d.ClassID == airplane_class_IDnum:
                    airplane_scores = d.Confidence
                    airplane_boxes = [[x1,y1,x2-x1,y2-y1]]
                    one_airplane_detected = one_airplane_detected+1


            # for ODclassid, ODscore, ODbox in zip(ODclasses.flatten(), ODconfs.flatten(), ODboxes):  # for issd in indices:
            #     # ODclassid = ODclasses[issd][0]
            #     # issd = issd[0]
            #     # ODscore = ODconfs[issd]
            #     # ODbox = ODboxes[issd]

            #     color = COLORS[int(ODclassid) % len(COLORS)]
            #     label = "%s : %s" % (ODclass_names[ODclassid - 1], round(ODscore * 100, 2))
            #     cv2.rectangle(frameRD, ODbox, color, 2)
            #     cv2.putText(frameRD, label, (ODbox[0], ODbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)


            # airplane_Idxs = np.where(ODclasses == [airplane_class_IDnum]) # airplane_class_IDnum = 4 : the 5th class is airplane in coco dataset
            # airplane_scores = ODconfs[airplane_Idxs]
            # airplane_boxes  = ODboxes[airplane_Idxs[0],:]
            #print("SCORE",airplane_scores)
            if one_airplane_detected == 1 and airplane_scores>AIRPLANE_DETECTION_CONF_BEFORE_TRACKING: # maybe 80% is a bit high, 75% could work too
                Trajectory = np.array([[0, 0]])
                Trajectory = np.append(Trajectory, np.array([[airplane_boxes[0][0], airplane_boxes[0][1]]]), axis=0)
                TailNumberLists = []
                TailNumberCharScoreLists = []

                tracker = cv2.TrackerMOSSE_create() # in opencv 4.5.1 we only have CSRT and KCF check the tracking API in opencv doc https://docs.opencv.org/4.5.1/d9/df8/group__tracking.html
                tracker.init(frame, tuple(airplane_boxes[0]))
                success2 = True
                BOX_AIRCRAFT = []
                while success2: # think about a way to get out of the loop if an opposite direction operation happens___________________and those with stuck and fixed TRbbox
                    f += 1
                    success, frame = cap.read()
                    if success == False:
                        continue
                    frameRD = np.array(frame) # frame for RECORD

                    if len(Trajectory)>10:
                        aircraftINFO = ''










                    ########################################################################################################################
                    ################################################### OCR ################################################################
                    ########################################################################################################################
                    imgCuda = jetson.utils.cudaFromNumpy(frame)
                    detections = ODnet.Detect(imgCuda, overlay = "OVERLAY_NONE")
                    if len(detections) != 0:
                        one_airplane_detected=0
                        airplane_scores = 0
                        for d in detections:
                            x1,y1,x2,y2= int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)                           
                            if d.ClassID == airplane_class_IDnum:
                                airplane_scores = d.Confidence
                                airplane_boxes = [[x1,y1,x2-x1,y2-y1]]
                                one_airplane_detected = one_airplane_detected+1
#                     ODclasses, ODconfs, ODboxes = ODnet.detect(frame, confThreshold=OD_CONFIDENCE_THRESHOLD, nmsThreshold=OD_NMS_THRESHOLD)
#                     if len(ODclasses) != 0:
#                         airplane_Idxs = np.where(ODclasses == [airplane_class_IDnum])  # airplane_class_IDnum = 4 : the 5th class is airplane in coco dataset
#                         airplane_scores = ODconfs[airplane_Idxs]
#                         airplane_boxes = ODboxes[airplane_Idxs[0], :]

                        if len(airplane_boxes) == 1 and airplane_scores > AIRPLANE_DETECTION_CONF_AFTER_TRACKING: #0.7
                            xAIRCRAFT, yAIRCRAFT, wAIRCRAFT, hAIRCRAFT = airplane_boxes[0][0], airplane_boxes[0][1], airplane_boxes[0][2], airplane_boxes[0][3]
                            boxAIRCRAFT = frame[yAIRCRAFT:yAIRCRAFT + hAIRCRAFT, xAIRCRAFT:xAIRCRAFT + wAIRCRAFT, :]
                            BOX_AIRCRAFT.append(boxAIRCRAFT)

# #######################################################################################################################################################
#                             # boxAIRCRAFT = cv2.GaussianBlur(boxAIRCRAFT, (5, 5), 10,10)
#                             boxAIRCRAFT = cv2.bilateralFilter(boxAIRCRAFT,10,75,75)
#                             startT1 = time.time()
#                             OCRBbox, OCRconfidence = TextRegionDetectorCNN.detect(boxAIRCRAFT)
#                             endT1 = time.time()
#                             print('text region detector')
#                             print(endT1 - startT1)
#                             if len(OCRBbox) != 0:
#                                 if OCRconfidence[0] > OCR_CONFIDENCE_THRESHOLD:
#                                     startT = time.time()

#                                     OCRx, OCRy, OCRw, OCRh = OCRBbox[0]
#                                     # x=x-10
#                                     # y=y-5
#                                     # w=w+20
#                                     # h=h+10

#                                     vertices = np.array([[OCRx, OCRy + OCRh],
#                                                          [OCRx, OCRy],
#                                                          [OCRx + OCRw, OCRy],
#                                                          [OCRx + OCRw, OCRy + OCRh]], dtype="float32")
#                                     cropped = fourPointsTransform(boxAIRCRAFT, vertices)
#                                     cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#                                     blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
#                                     OCRrecognizer.setInput(blob)
#                                     result = OCRrecognizer.forward()
#                                     wordRecognized, charScore_list = decodeText2(result)
#                                     endT = time.time()
#                                     print('text recognition')
#                                     print(endT - startT)

#                                     for j in range(4):
#                                         p1 = (vertices[j][0] + np.array(xAIRCRAFT, dtype="float32"), vertices[j][1] + np.array(yAIRCRAFT, dtype="float32")) # np.array(xAIRCRAFT, dtype="float32") bcz of aircraft box
#                                         p2 = (vertices[(j + 1) % 4][0] + np.array(xAIRCRAFT, dtype="float32"), vertices[(j + 1) % 4][1] + np.array(yAIRCRAFT, dtype="float32"))
#                                         cv2.line(frameRD, p1, p2, (0, 255, 255), 1)
#                                     cv2.putText(frameRD, 'Tail Number Detected', (int(vertices[1][0]+xAIRCRAFT), int(vertices[1][1]+yAIRCRAFT)),
#                                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                                     cv2.putText(frameRD, 'Aircraft Tail Number:' + wordRecognized.upper(), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
#                                     TailNumberLists.append(wordRecognized.upper())
#                                     TailNumberCharScoreLists.append(charScore_list)

                    ########################################################################################################################
                    ################################################### OCR ################################################################
                    ########################################################################################################################

                    startTR = time.time()
                    success2, TRbbox = tracker.update(frame)
                    endTR = time.time()
                    print('tracker')
                    print(endTR - startTR)

                    if success2 == False or TRbbox[0]<0 or (TRbbox[0]+TRbbox[2])>frame_width:
                        cut_in_traffic = True
                        put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)
                        break
                    if (len(Trajectory)>3):
                        if (Trajectory[-1,0]-Trajectory[-2,0]==0)*(Trajectory[-2,0]-Trajectory[-3,0]==0):
                            cut_in_traffic = True
                            put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)
                            break

                    ii[0:2] = ii[1:3]
                    ii[2] = 1
                    Text1 = 'Status : New Operation in Track'
                    x, y, w, h = int(TRbbox[0]), int(TRbbox[1]), int(TRbbox[2]), int(TRbbox[3])
                    cv2.rectangle(frameRD, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
                    cv2.putText(frameRD, 'AIRCRAFT DETECTED', (int(TRbbox[0]), int(TRbbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)
                    Trajectory = np.append(Trajectory, np.array([[TRbbox[0], TRbbox[1]]]), axis=0)  # [x,y] = bboxs[0,:]___work on it to give the best fit box possible
                ii[0:2] = ii[1:3]
                ii[2] = 0
                #Text1 = 'Status : No Operation is Going On'
        else: #len(ODclasses) == 0: no object detected
            put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)

    else:
        ii[2] = 0
        if sum(ii) == 0:
            Text1 = 'Status : No Operation is Going On'
        put_Text_and_Record(frameRD, Text1, Text2, Text3, Text4, aircraftINFO, database, out)


    ####################################################################################################################
    ########################################### Trajectory Processing ##################################################
    ####################################################################################################################

    # CAMERA LEFT TO RIGHT
    if cut_in_traffic:  # sum(ii) == 0: # means cut in traffic
        cut_in_traffic = False
        if np.size(Trajectory) > 2:  # means Trajectory is formed
            Trajectory = np.delete(Trajectory, 0, 0)  # to delete the unwanted first row
            T.append(Trajectory)

            if len(Trajectory) > 4 and len(Trajectory) <= 10:
                #if (frame_height-Trajectory[0,1])>(frame_height/2):
                    speed = -(Trajectory[0, 0] - Trajectory[-1, 0]) / len(Trajectory)
                    if sum((Trajectory[-5:, 0] + TRbbox[2]) > np.array([frame_width - 100, frame_width - 100, frame_width - 100, frame_width - 100,frame_width - 100])) > 1 and speed > 3:  # np.std(Trajectory[:,1])<20 and     add a y threshold constraint too
                        if TRbbox[2] * TRbbox[3] < AreaLandings:
                            L += 1
                            Text2 = 'Last Operation: ' + ' LANDING'  # Text2 = 'Last Operation: ' + finalized_tail_number(TailNumberLists)[0] + ' LANDED'
                            # aircraftINFO = finalized_tail_number(TailNumberLists)[1]
                            # if len(aircraftINFO) == 20:  # means aircraftINFO=='Not in Utah Database':
                            #     aircraftINFO = finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists)[1]
                            Text4 = '# Landings   : ' + str(L)
                            save_operation_files(text=Text4, BOX_AIRCRAFT=BOX_AIRCRAFT)

            # Trim the Trajectory
            if len(Trajectory) > 10:  # ONLY FOR SYS 2##############
                speed = -(Trajectory[0, 0] - Trajectory[-1, 0]) / len(Trajectory)

                if sum((Trajectory[-5:, 0] + TRbbox[2]) > np.array([frame_width - 100, frame_width - 100, frame_width - 100, frame_width - 100,frame_width - 100])) > 1 and speed > 3:  # np.std(Trajectory[:,1])<20 and     add a y threshold constraint too
                    if TRbbox[2] * TRbbox[3] < AreaLandings and (frame_height-Trajectory[0,1])>((1/3)*frame_height):
                        L += 1
                        Text2 = 'Last Operation: ' + ' LANDING'  # Text2 = 'Last Operation: ' + finalized_tail_number(TailNumberLists)[0] + ' LANDED'
                        # aircraftINFO = finalized_tail_number(TailNumberLists)[1]
                        # if len(aircraftINFO) == 20:  # means aircraftINFO=='Not in Utah Database':
                        #     aircraftINFO = finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists)[1]
                        Text4 = '# Landings   : ' + str(L)
                        save_operation_files(text=Text4, BOX_AIRCRAFT=BOX_AIRCRAFT)
                    else:
                        D += 1
                        Text2 = 'Last Operation: ' + ' DEPARTURE'  # Text2 = 'Last Operation: ' + finalized_tail_number(TailNumberLists)[0] + ' DEPARTED'
                        # aircraftINFO = finalized_tail_number(TailNumberLists)[1]
                        # if len(aircraftINFO) == 20:  # means aircraftINFO=='Not in Utah Database':
                        #     aircraftINFO = finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists)[1]
                        Text3 = '# Departures : ' + str(D)
                        save_operation_files(text=Text3, BOX_AIRCRAFT=BOX_AIRCRAFT)
                if sum(Trajectory[-5:, 0] < np.array([100, 100, 100, 100, 100])) > 1 and speed < -3:  # np.std(Trajectory[:,1])>10 and         add y point1 > nesf ghade ax_______________
                    L += 1
                    Text2 = 'Last Operation: '  + ' ARRIVAL'#Text2 = 'Last Operation: ' + finalized_tail_number(TailNumberLists)[0] + ' LANDED'
                    # aircraftINFO = finalized_tail_number(TailNumberLists)[1]
                    # if len(aircraftINFO) == 20: # means aircraftINFO=='Not in Utah Database':
                    #     aircraftINFO = finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists)[1]
                    Text4 = '# Landings   : ' + str(L)
                    save_operation_files(text=Text4, BOX_AIRCRAFT=BOX_AIRCRAFT)
            Trajectory = np.array([[0, 0]])
            i = -1
            

######### after a post processing in the actual implementation (not the recorded video, put a criteria for stop
# processong the next possible operation for about 5 second to avoid double counting bcz of the end detections #########

    if cv2.waitKey(40) == 27:
        break
    #if cv2.waitKey(1) == ord('q'):
    #    break
cv2.destroyAllWindows()
cap.release()
out.release()



# box_area = TRbbox[2] * TRbbox[3]
# tx0 = frame_width - (Trajectory[0,0]+TRbbox[2]) if right  = Trajectory[0,0] if left
# ave_ty = (Trajectory[0,1]+Trajectory[-1,1])/2+TRbbox[3]