import cv2
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow import keras
################################################################
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
#################################################################
# https://docs.opencv.org/4.4.0/db/d28/tutorial_cascade_classifier.html
# https://docs.opencv.org/4.4.0/dc/d88/tutorial_traincascade.html

number_skipped_frames = 5
#cap = cv2.VideoCapture("D:\Aircraft detection project\Data Collection/10-30-2020 skypark/1080 2x 160ft from runway/GH013783.mp4")
numberoftopclasses_for_identification = 8 # number of top aircraft classes considered at the end to filter the database

topKclass = 5 # number of top aircraft classes considered at each frame

# number of tail number recognitions per frame:
topKtn_per_frame = 3 # you should change it on Line 326
#################################################                                                                           CHECH Database for County and states


def empty(a):
    pass

########################################################################################################################
################################################### PARAMETERS #########################################################
########################################################################################################################
# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",30,1000,empty)
cv2.createTrackbar("Neig","Result",4,50,empty) ####################### USE the refinement filter proposed in the I3CE paper
cv2.createTrackbar("Min Area","Result",1*15,100000,empty)
cv2.createTrackbar("Max Area","Result",450*45,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

yolo_confidence = 0.6
ocr_confidence = 0.3
Cextension = 4
refineTHRESH = 0.01
one = 1


########################################################################################################################
################################################### PARAMETERS #########################################################
########################################################################################################################
# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)
########################################### Classification Model #######################################################
model = keras.models.load_model("Aircraft Classification/best-weights-171-0.75")
IGclass_names = ['a single piston',
                 'b single turboprop',
                 'c multi turboprop w lower20',      # detection of this class activates "multi pistons" too
                 'd multi turboprop w higher20',
                 'e multi turbofan w lower20 2engine',
                 'f multi turbofan w lower20 3engine',
                 'g multi turbofan w higher20 2engines',
                 'h multi turbofan w higher20 2engines AIRBUS',
                 'i multi turbofan w higher20 2engines BOEING',
                 'j multi turbofan w higher20 2engines GULFSTREAM',
                 'k multi turbofan w higher20 2engines MCDONNELDOUGLAS',
                 'l multi turbofan w higher20 3engines',
                 'm multi turbofan w higher20 4engines']
#################################################### IoU ###############################################################
def I_o_U(boxDetection, bboxTRACKER):
    if boxDetection[3]>0: # we have a detection
    # Detection coordinates
        x1D = boxDetection[0]
        y1D = boxDetection[1]
        x2D = boxDetection[0] + boxDetection[2]
        y2D = boxDetection[1] + boxDetection[3]
        DetectionAREA = boxDetection[2]*boxDetection[3]
    # TRACKER coordinates
        x1T = bboxTRACKER[0]
        y1T = bboxTRACKER[1]
        x2T = bboxTRACKER[0] + bboxTRACKER[2]
        y2T = bboxTRACKER[1] + bboxTRACKER[3]
        GroundTruthAREA = bboxTRACKER[2]*bboxTRACKER[3]
    # intersection coordinates
        x1I = max(x1D, x1T)
        y1I = max(y1D, y1T)
        x2I = min(x2D, x2T)
        y2I = min(y2D, y2T)
        IntersectionAREA= max(0, x2I - x1I + 1) * max(0, y2I - y1I + 1)

        IoU = IntersectionAREA / float(DetectionAREA + GroundTruthAREA - IntersectionAREA)
    else:
        IoU = 0
    return IoU
########################################################################################################################

##################################### object detection #################################################################
# learned from: https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
ODclass_names= []
ODclassFile = 'Models/coco.names'
with open(ODclassFile,'rt') as f:
    ODclass_names = f.read().rstrip('\n').split('\n')
airplane_class_IDnum = 4
# pip install opencv-python #version 4.4.0.46
ODnet = cv2.dnn.readNet("Models/yolov4.weights", "Models/yolov4.cfg")
ODnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ODnet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ODmodel = cv2.dnn_DetectionModel(ODnet)
ODmodel.setInputParams(size=(416, 416), scale=1/255)
########################################################################################################################
################################################### OCR ################################################################
########################################################################################################################
def decodeText2(scores):
    text = ""
    alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    charScore_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
            charScore_list.append(scores[i])
    return ''.join(char_list), charScore_list

def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

modelArchFilename = 'Detector Models/textbox.prototxt'
modelWeightsFilename = 'Detector Models/TextBoxes_icdar13.caffemodel'
TextRegionDetectorCNN	=	cv2.text.TextDetectorCNN_create(	modelArchFilename, modelWeightsFilename	)

OCRmodelRecognition = "Recognition Models/CRNN_VGG_BiLSTM_CTC.onnx"
OCRrecognizer = cv2.dnn.readNetFromONNX(OCRmodelRecognition)

########################################################################################################################
########################################################################################################################
########################################################################################################################
def database_filter(aircraft_class_number): # aircraft_class_number: 0 1 2 3 4 5 6 7 8 9 10 11 12
    if aircraft_class_number==0:
        ACN = (database['TYPE AIRCRAFT'] == '4') * ((database['TYPE ENGINE'] == 1)+(database['TYPE ENGINE'] == 7)+(database['TYPE ENGINE'] == 8)+(database['TYPE ENGINE'] == 11)) * (database['AC-weight']=='CLASS 1')
    if aircraft_class_number == 1:
        ACN = (database['TYPE AIRCRAFT'] == '4') * (database['TYPE ENGINE'] == 2) * (database['AC-weight'] == 'CLASS 1')
    if aircraft_class_number == 2:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 2)+(database['TYPE ENGINE'] == 1)+(database['TYPE ENGINE'] == 7)+(database['TYPE ENGINE'] == 8)+(database['TYPE ENGINE'] == 11)) * ((database['AC-weight'] == 'CLASS 1')+(database['AC-weight'] == 'CLASS 2'))
    if aircraft_class_number == 3:
        ACN = (database['TYPE AIRCRAFT'] == '5') * (database['TYPE ENGINE'] == 2) * (database['AC-weight'] == 'CLASS 3')
    if aircraft_class_number == 4:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * ((database['AC-weight'] == 'CLASS 1')+(database['AC-weight'] == 'CLASS 2')) * (database['NO-eng'] == 2)
    if aircraft_class_number == 5:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * ((database['AC-weight'] == 'CLASS 1')+(database['AC-weight'] == 'CLASS 2')) * (database['NO-eng'] == 3)
    if aircraft_class_number == 6:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 2)
    if aircraft_class_number == 7:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 2)# * (database['Mfr'] == 'AIRBUS')
    if aircraft_class_number == 8:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 2)# * (database['Mfr'] == 'BOEING')
    if aircraft_class_number == 9:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 2)# * (database['Mfr'] == 'GULFSTREAM')
    if aircraft_class_number == 10:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 2)# * (database['Mfr'] == 'MCDONNELL DOUGLAS')
    if aircraft_class_number == 11:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 3)
    if aircraft_class_number == 12:
        ACN = (database['TYPE AIRCRAFT'] == '5') * ((database['TYPE ENGINE'] == 4)+(database['TYPE ENGINE'] == 5)) * (database['AC-weight'] == 'CLASS 3') * (database['NO-eng'] == 4)
    return ACN

def database_maker(aircraft_class_number_list):
    for i in range(len(aircraft_class_number_list)):
        if i == 0:
            ACN = database_filter(aircraft_class_number_list[i])
        if i > 0:
            ACN = database_filter(aircraft_class_number_list[i]) + ACN
    return ACN

list_top_classes = []
########################################################################################################################
###################################### joint analysis 1 (maximum likelihood) ############################################
########################################################################################################################

from collections import OrderedDict 
import pandas as pd

my_database =pd.read_pickle("Aircraft Identification\MY_MASTER3_UtahState.pkl") # pure number tail numbers are imported as string   &&   '100  ' ---> '100'
# bountiful airport neighborhood counties: davis 011 (itself) + salt lake 035 + weber 057 + morgan 029 + tooele 045 + box elder 003
# brigham airport neighborhood counties: box elder 003 (itself) + cache 005 + weber 057 + davis 011 + tooele 045
# For your videos and images, you need:      washington 053 + davis + salt lake + utah + sevier 041 + weber
database = my_database[(my_database['COUNTY']=='011') + (my_database['COUNTY']=='035') + (my_database['COUNTY']=='057') + (my_database['COUNTY']=='029') + (my_database['COUNTY']=='045') + (my_database['COUNTY']=='003')+
                                      (my_database['COUNTY']=='005')+
                                      (my_database['COUNTY']=='053') + (my_database['COUNTY']=='041')]
database = database.reset_index()
database1 = database
database1.drop('level_0', inplace=True, axis=1)

# database = pd.read_excel ('D:\Aircraft detection project\python\SYSTEM 2 operation count TN det_textbox_rec_crnn faster yolo/Database.xls', sheet_name='allcountiesaddedhere')     ######################################################################### Do sth and include all counties
# for irn in range(len(database)): # we need this bcz pure number tail numbers (like N65474 or 65474) are imported as number(float) and not string
#     database['n_number'][irn] = str(database['n_number'][irn])
TailNumberListsHaar = []
TailNumberListsTextBoxes = []
TailNumberCharScoreListsHaar = []
TailNumberCharScoreListsTextBoxes = []

def finalized_tail_number(TNlists, aircraft_class_number_list=None):
    if aircraft_class_number_list != None:
        ACN = database_maker(aircraft_class_number_list)
        database2 = database1[ACN]
        #database2.drop('level_0', inplace=True, axis=1)
        database2 = database2.reset_index()
    else:
        database2 = database1

    sorted_TNlists = sorted(TNlists, key=TNlists.count, reverse=True)  # sort by frequency
    shortlist = list(OrderedDict.fromkeys(sorted_TNlists))  # duplicates are removed
    JPA0 = 'Unrecognized Aircraft'
    aircraft_identified = False
    for tln in shortlist:
        if len(tln)>1:
            JPA0 = shortlist[0]
            if database2[database2['N-NUMBER'] == tln[1:]].size == 0: # this tln (tail number) is not in the database
                continue
            else: # this tln will be selected as the finalized tail number
                index = database2[database2['N-NUMBER'] == tln[1:]].index.values[0]
                aircraft_detailed_info = database2.loc[index]
                finalized_TN = tln
                aircraft_identified = True
                break
    if aircraft_identified == False and len(shortlist) != 0:
        aircraft_detailed_info = 'Not in Database'
        finalized_TN = 'go with JPA2'#shortlist[0]
    if aircraft_identified == False and len(shortlist) == 0:
        aircraft_detailed_info = ''
        finalized_TN = 'Unrecognized Aircraft'
    return finalized_TN, aircraft_detailed_info, JPA0


########################################################################################################################
#########################################        joint analysis 2       ###############################################
########################################################################################################################
def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

Alphabet = "0123456789abcdefgh1jklmn0pqrstuvwxyz"

def jointProbabilityDecoder(Det_tailnumber, charScore_list, aircraft_class_number_list=None): #Det_tailnumber must be in lowercase letters like 'n120bf'
    if aircraft_class_number_list != None:
        ACN = database_maker(aircraft_class_number_list)
        database2 = database1[ACN]
        #database2.drop('level_0', inplace=True, axis=1)
        database2 = database2.reset_index()
    else:
        database2 = database1

    if len(Det_tailnumber) < 2 or len(Det_tailnumber) > 8:
        most_probable_TN = ''
        aircraft_detailed_info = 'Error'
        most_probable_TNs = []
    else:
        if len(Det_tailnumber) == 8: # makes it 6
            Det_tailnumber = Det_tailnumber[1:7]     #drop the first and the last character
            charScore_list = charScore_list[1:7]
        if len(Det_tailnumber) == 7: # makes it 6
            if Det_tailnumber[0].lower() == 'n' and Det_tailnumber[0].lower() != 'n':
                Det_tailnumber = Det_tailnumber[0:6] #drop the last character
                charScore_list = charScore_list[0:6]
            else:
                Det_tailnumber = Det_tailnumber[1:7] #drop the first character
                charScore_list = charScore_list[1:7]

        Det_tailnumber = Det_tailnumber[1:]          # DROP 'n' since the databse n_numbers do not have 'N'
        charScore_list = charScore_list[1:]
        charSoftProb = []
        for i in range(len(Det_tailnumber)):
            charSoftProb.append(softmax(charScore_list[i]))

        probabilities = np.zeros(len(database2['N-NUMBER']))
        for index, N_number in enumerate(database2['N-NUMBER']):
            if len(Det_tailnumber) == len(N_number):
                probability = 0
                for ind, char in enumerate(N_number): # N_number is in uppercase
                    if Det_tailnumber[ind] == char.lower():
                        #probability += 1 ######### WHY??? why dont we use the actual probability######################################################
                        c = Alphabet.find(char.lower())
                        probability = probability + charSoftProb[ind][0][c + 1]
                    else:
                        c = Alphabet.find(char.lower())
                        probability = probability + charSoftProb[ind][0][c+1]

                probabilities[index]=probability/len(N_number)

        if sum(probabilities) == 0: # no N_number found with the same length
            most_probable_TN = ''
            aircraft_detailed_info = 'Most Likely Not in Database'
            most_probable_TNs = []
        else:
            most_probable_TN_index = np.argmax(probabilities)
            aircraft_detailed_info = database2.loc[most_probable_TN_index]
            most_probable_TN = 'N' + database2['N-NUMBER'][most_probable_TN_index]

            # topKtn_per_frame = 3
            most_probable_TNs = []
            for tnpf in range(topKtn_per_frame):
                most_probable_TNs.append('N' + database2['N-NUMBER'][np.argsort(probabilities)[-(tnpf + 1)]])

            # Second_most_probable_TN_index = np.argsort(probabilities)[-2]
            # Second_most_probable_TNs = 'N' + database2['N-NUMBER'][Second_most_probable_TN_index]
            # Third_most_probable_TN_index  = np.argsort(probabilities)[-3]
            # Third_most_probable_TNs = 'N' + database2['N-NUMBER'][Third_most_probable_TN_index]
            # most_probable_TNs = [most_probable_TN, Second_most_probable_TNs, Third_most_probable_TNs]

    return most_probable_TN, aircraft_detailed_info, most_probable_TNs

def finalized_tail_number2(TailNumberLists, TailNumberCharScoreLists, aircraft_class_number_list=None):
    if aircraft_class_number_list != None:
        ACN = database_maker(aircraft_class_number_list)
        database2 = database1[ACN]
        #database2.drop('level_0', inplace=True, axis=1)
        database2 = database2.reset_index()
    else:
        database2 = database1

    most_probable_TN_List = []
    aircraft_detailed_info_List = []
    most_probable_TN_List_TopK = []
    shortlist_sorted_TNlists_TopK = []
    for i in range(len(TailNumberLists)):
        most_probable_TN, aircraft_detailed_info, TopK_most_probable_TNs = jointProbabilityDecoder(TailNumberLists[i], TailNumberCharScoreLists[i], aircraft_class_number_list)
        if most_probable_TN != '':
            most_probable_TN_List.append(most_probable_TN)
            aircraft_detailed_info_List.append(aircraft_detailed_info)

            topKtn_per_frame = len(TopK_most_probable_TNs)
            for ktnpf in range(topKtn_per_frame):
                most_probable_TN_List_TopK.append(TopK_most_probable_TNs[ktnpf])
    if len(most_probable_TN_List) == 0:
        finalized_TN = ''
        aircraft_detailed_infoo = 'Most Likely Not in Database'
        finalized_TN_TopK = ''
    else:
        sorted_TNlists = sorted(most_probable_TN_List, key=most_probable_TN_List.count, reverse=True)
        finalized_TN = sorted_TNlists[0]
        index = database2[database2['N-NUMBER'] == finalized_TN[1:]].index.values[0]
        aircraft_detailed_infoo = database2.loc[index]
        # aircraft_detailed_infoo = aircraft_detailed_info_List[0]
        sorted_TNlists_TopK = sorted(most_probable_TN_List_TopK, key=most_probable_TN_List_TopK.count, reverse=True)
        finalized_TN_TopK = sorted_TNlists_TopK[0]
        shortlist_sorted_TNlists_TopK = list(OrderedDict.fromkeys(sorted_TNlists_TopK))  # duplicates are removed
    return finalized_TN, aircraft_detailed_infoo, finalized_TN_TopK, shortlist_sorted_TNlists_TopK
########################################################################################################################
########################################################################################################################
########################################################################################################################
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f = 0
while f < length:
    f += 1
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)
    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    for skip in range(number_skipped_frames):
        success, img = cap.read()
    #img = cv2.flip(img, 1)                                                        # for LEFT collections
    if success==False:
        continue
    #cv2.imwrite("D:\Aircraft detection project\Aircraft Identification\DATA PAPER aircraft identification/AC_4_4_{}.jpg".format(f), img)
    ########################################################################################################################
    ########################################### aircraft detection #########################################################
    ########################################################################################################################
    startOD = time.time()
    ODclasses, ODscores, ODboxes = ODmodel.detect(img, yolo_confidence, 0.4)#OD_CONFIDENCE_THRESHOLD = 0.6, OD_NMS_THRESHOLD = 0.4
    endOD = time.time()
    print("YOLO detection = ", endOD - startOD)
    if len(ODclasses) != 0:
        airplane_Idxs = np.where(ODclasses == [airplane_class_IDnum])  # airplane_class_IDnum = 4 : the 5th class is airplane in coco dataset
        airplane_scores = ODscores[airplane_Idxs]
        airplane_boxes = ODboxes[airplane_Idxs[0], :]

        if len(airplane_boxes) == 1 and airplane_scores > yolo_confidence:  # 0.7  ############# AIRPLANE_DETECTION_CONF_AFTER_TRACKING
            xAIRCRAFT, yAIRCRAFT, wAIRCRAFT, hAIRCRAFT = airplane_boxes[0][0], airplane_boxes[0][1], airplane_boxes[0][2], airplane_boxes[0][3]
            boxAIRCRAFT = img[yAIRCRAFT:yAIRCRAFT + hAIRCRAFT, xAIRCRAFT:xAIRCRAFT + wAIRCRAFT, :]
 ###############################################################################################################################
######################################## Image Classification #####################################################################
 ###############################################################################################################################
            yL,yR,xL,xR=50,50,50,50
            if xAIRCRAFT<xL:
                xL=xAIRCRAFT
            if yAIRCRAFT < yL:
                yL = yAIRCRAFT
            boxAIRCRAFT_extended_for_classification = img[yAIRCRAFT-yL:yAIRCRAFT + hAIRCRAFT+yR, xAIRCRAFT-xL:xAIRCRAFT + wAIRCRAFT+xR, :]
            cv2.imwrite('temp/aircraft_box.jpg', boxAIRCRAFT_extended_for_classification)
            img_path = 'temp/aircraft_box.jpg'
            xxx = image.load_img(img_path, target_size=(256, 256))
            # img = cv2.resize(box, (224, 224))
            xx = image.img_to_array(xxx)
            xx = np.expand_dims(xx, axis=0)
            xx = preprocess_input(xx)

            preds = model.predict(xx)
            # print('Predicted:', decode_predictions(preds, top=3)[0])
            print('Predicted:', preds)
            print(np.argmax(preds, axis=-1))  # 0 1 2 3 4 5 6 7 8 9 10 11 12
            top_classes = np.argsort(-preds, axis=-1)[0][0:topKclass]
            print('1st class', IGclass_names[top_classes[0]])
            print('2nd class', IGclass_names[top_classes[1]])
            print('3rd class', IGclass_names[top_classes[2]])

            for ltc in range(topKclass):
                list_top_classes.append(list(top_classes)[ltc])
###############################################################################################################################
###############################################################################################################################
            cv2.rectangle(img, (xAIRCRAFT, yAIRCRAFT), (xAIRCRAFT + wAIRCRAFT, yAIRCRAFT + hAIRCRAFT), (255, 0, 0), 4)
            # boxAIRCRAFT = cv2.bilateralFilter(boxAIRCRAFT, 10, 75, 75)
            startT1 = time.time()
            OCRBboxTextBoxes, OCRconfidenceTextBoxes = TextRegionDetectorCNN.detect(boxAIRCRAFT)
            endT1 = time.time()
            print('TextBoxes detector', endT1 - startT1)



            gray = cv2.cvtColor(boxAIRCRAFT, cv2.COLOR_BGR2GRAY)
            # claheDst = cv2.Mat()
            # cv2.imshow("before CLAHE", gray)
            # cv2.waitKey(0)
            clahe = cv2.createCLAHE(2, (8, 8))
            gray = clahe.apply(gray)
            # cv2.imshow("after CLAHE", gray)
            # cv2.waitKey(0)
            # clahe.delete()


            # DETECT THE OBJECT USING THE CASCADE
            scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
            neig=cv2.getTrackbarPos("Neig", "Result")

            start = time.time()
            objects = cascade.detectMultiScale(gray,scaleVal, neig)
            ############################################# refinement 1 #############################################################
            if len(objects) > one:
                while True:
                    L = len(objects)
                    intersects = np.array([[-1, -1]])
                    i = -1
                    for (x1, y1, w1, h1) in objects:
                        i = i + 1
                        box1 = np.array([x1, y1, Cextension*w1, h1])
                        j = -1
                        for (x2, y2, w2, h2) in objects:
                            j = j + 1
                            box2 = np.array([x2, y2, Cextension*w2, h2])
                            if I_o_U(box1, box2) > refineTHRESH and i < j:
                                intersects = np.append(intersects, np.array([[i, j]]), axis=0)
                    intersects = np.delete(intersects, 0, 0)

                    for (i, j) in intersects:
                        objects[j] = np.array([0, 0, 0, 0])
                    objects = objects[~np.all(objects == 0, axis=1)]

                    if L == len(objects):
                        break
            ########################################################################################################################
            end = time.time()
            #print('Haar')
            print("N detection = ",end - start)

            # DISPLAY THE DETECTED OBJECTS
            Haardetected = False
            for (x, y, w, h) in objects:
                x = x + xAIRCRAFT
                y = y + yAIRCRAFT
                area = w * h
                minArea = cv2.getTrackbarPos("Min Area", "Result")
                maxArea = cv2.getTrackbarPos("Max Area", "Result")
                if area > minArea and area < maxArea:
                    cv2.rectangle(img, (x, y), (x + w * Cextension, y + h), color, 5)
                    cv2.putText(img, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
                    roi_color = img[y:y + h, x:x + 6 * w]
                    OCRBboxHaar = np.array([[x - xAIRCRAFT, y - yAIRCRAFT, Cextension * w, h]])
                    Haardetected = True
            # if objects != ():

    ########################################################################################################################
    ################################################### OCR ################################################################
    ########################################################################################################################

            for ocrboxindex in range(2):
                    OCRBbox = []
                    if ocrboxindex == 0 and Haardetected:
                        OCRBbox = OCRBboxHaar
                    if ocrboxindex == 1 and OCRconfidenceTextBoxes[0] > ocr_confidence:
                        OCRBbox = OCRBboxTextBoxes

                    if len(OCRBbox) != 0:
                        if True:  # OCRconfidence[0] > OCR_CONFIDENCE_THRESHOLD:

                            OCRx, OCRy, OCRw, OCRh = OCRBbox[0]
                            # x=x-10
                            # y=y-5
                            # w=w+20
                            # h=h+10

                            vertices = np.array([[OCRx, OCRy + OCRh],
                                                 [OCRx, OCRy],
                                                 [OCRx + OCRw, OCRy],
                                                 [OCRx + OCRw, OCRy + OCRh]], dtype="float32")
                            startTR = time.time()
                            cropped = fourPointsTransform(boxAIRCRAFT, vertices)
                            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                            # cv2.imshow( "blob", cropped)
                            # cv2.waitKey(0)
                            blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
                            # cv2.imshow(blob, "blob")
                            OCRrecognizer.setInput(blob)
                            result = OCRrecognizer.forward()
                            wordRecognized, charScore_list = decodeText2(result)
                            #wordRecognized = decodeText(result)
                            endTR = time.time()
                            print("OCR detection = ", endTR - startTR)

                            if ocrboxindex == 1:  # OCRboxTextboxes
                                for j in range(4):
                                    p1 = (vertices[j][0] + np.array(xAIRCRAFT, dtype="float32"),
                                          vertices[j][1] + np.array(yAIRCRAFT,
                                                                    dtype="float32"))  # np.array(xAIRCRAFT, dtype="float32") bcz of aircraft box
                                    p2 = (vertices[(j + 1) % 4][0] + np.array(xAIRCRAFT, dtype="float32"),
                                          vertices[(j + 1) % 4][1] + np.array(yAIRCRAFT, dtype="float32"))
                                    cv2.line(img, p1, p2, (0, 255, 255), 3)
                                cv2.putText(img, 'Tail Number Detected',
                                            (int(vertices[1][0] + xAIRCRAFT), int(vertices[1][1] + yAIRCRAFT)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                            if ocrboxindex == 0:
                                cv2.putText(img, 'Aircraft Tail Number (Haar):' + wordRecognized.upper(),
                                            (50, ocrboxindex * 50 + 250),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
                                TailNumberListsHaar.append(wordRecognized.upper())
                                TailNumberCharScoreListsHaar.append(charScore_list)
                            else:
                                cv2.putText(img, 'Aircraft Tail Number (TextBoxes):' + wordRecognized.upper(),
                                            (50, ocrboxindex * 50 + 250),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 250, 0), 4)
                                TailNumberListsTextBoxes.append(wordRecognized.upper())
                                TailNumberCharScoreListsTextBoxes.append(charScore_list)

    ########################################################################################################################
    ################################################### OCR ################################################################
    ########################################################################################################################

    frames = cv2.resize(img, (1080, 608))
    cv2.imshow("Result", frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break


JPA1_Haar, placeholder, JPA0_Haar =           finalized_tail_number(TailNumberListsHaar)
JPA1_TextBoxes, placeholder, JPA0_TextBoxes = finalized_tail_number(TailNumberListsTextBoxes)
print("finalized_tail_number JPA0 (Haar): ", JPA0_Haar) #finalized_tail_number(TailNumberListsHaar)[2])
print("finalized_tail_number JPA0 (TextBoxes): ", JPA0_TextBoxes)#finalized_tail_number(TailNumberListsTextBoxes)[2])
print("finalized_tail_number JPA1 (Haar): ", JPA1_Haar)#finalized_tail_number(TailNumberListsHaar)[0])
print("finalized_tail_number JPA1 (TextBoxes): ", JPA1_TextBoxes)#finalized_tail_number(TailNumberListsTextBoxes)[0])

JPA2_Haar, placeholder, JPA2_TopK_tailnumbers_per_frame_Haar, placeholder1 =           finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar)
JPA2_TextBoxes, placeholder, JPA2_TopK_tailnumbers_per_frame_TextBoxes, placeholder1 = finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes)
print("finalized_tail_number JPA2 (Haar): ", JPA2_Haar)#finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar)[0])
print("finalized_tail_number JPA2 (TextBoxes): ", JPA2_TextBoxes)#finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes)[0])
print("finalized_tail_number JPA2_TopK_tailnumbers_per_frame (Haar): ", JPA2_TopK_tailnumbers_per_frame_Haar)#finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar)[2])
print("finalized_tail_number JPA2_TopK_tailnumbers_per_frame (TextBoxes): ", JPA2_TopK_tailnumbers_per_frame_TextBoxes)#finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes)[2])


from collections import Counter
most_common = [key for key, val in Counter(list_top_classes).most_common(numberoftopclasses_for_identification)]
aircraft_class_number_list = most_common
print("aircraft_class_number_list:", aircraft_class_number_list)
JPA1_Haar_CLASSIFICATION = finalized_tail_number(TailNumberListsHaar, aircraft_class_number_list)[0]
JPA1_TextBoxes_CLASSIFICATION = finalized_tail_number(TailNumberListsTextBoxes, aircraft_class_number_list)[0]
print("finalized_tail_number JPA1 (Haar) + CLASSIFICATION: ", JPA1_Haar_CLASSIFICATION)#finalized_tail_number(TailNumberListsHaar, aircraft_class_number_list)[0])
print("finalized_tail_number JPA1 (TextBoxes) + CLASSIFICATION: ", JPA1_TextBoxes_CLASSIFICATION)#finalized_tail_number(TailNumberListsTextBoxes, aircraft_class_number_list)[0])
JPA2_Haar_CLASSIFICATION, placeholder, JPA2_TopK_tailnumbers_per_frame_Haar_CLASSIFICATION, placeholder1 = finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar, aircraft_class_number_list)
JPA2_TextBoxes_CLASSIFICATION, placeholder, JPA2_TopK_tailnumbers_per_frame_TextBoxes_CLASSIFICATION, most_probable_TN_List_TopK_TextBoxes = finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes, aircraft_class_number_list)
print("finalized_tail_number JPA2 (Haar) + CLASSIFICATION: ", JPA2_Haar_CLASSIFICATION)#finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar, aircraft_class_number_list)[0])
print("finalized_tail_number JPA2 (TextBoxes) + CLASSIFICATION: ", JPA2_TextBoxes_CLASSIFICATION)#finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes, aircraft_class_number_list)[0])
print("finalized_tail_number JPA2_TopK_tailnumbers_per_frame (Haar) + CLASSIFICATION: ", JPA2_TopK_tailnumbers_per_frame_Haar_CLASSIFICATION)#finalized_tail_number2(TailNumberListsHaar, TailNumberCharScoreListsHaar, aircraft_class_number_list)[0])
print("finalized_tail_number JPA2_TopK_tailnumbers_per_frame (TextBoxes) + CLASSIFICATION: ", JPA2_TopK_tailnumbers_per_frame_TextBoxes_CLASSIFICATION)#finalized_tail_number2(TailNumberListsTextBoxes, TailNumberCharScoreListsTextBoxes, aircraft_class_number_list)[0])


print('Haar: ',TailNumberListsHaar)
print('TextBoxes: ',TailNumberListsTextBoxes)
#print("number_of_detected_aircraft = ", number_of_detected_aircraft)

print(JPA0_Haar+','+ JPA0_TextBoxes+','+ JPA1_Haar+','+ JPA1_TextBoxes+','+ JPA2_Haar,','+ JPA2_TextBoxes+','+ JPA2_TopK_tailnumbers_per_frame_Haar+','+ JPA2_TopK_tailnumbers_per_frame_TextBoxes+','+
      JPA1_Haar_CLASSIFICATION+','+ JPA1_TextBoxes_CLASSIFICATION+','+ JPA2_Haar_CLASSIFICATION+','+ JPA2_TextBoxes_CLASSIFICATION+','+ JPA2_TopK_tailnumbers_per_frame_Haar_CLASSIFICATION+','+ JPA2_TopK_tailnumbers_per_frame_TextBoxes_CLASSIFICATION)

print("List most_probable_TN_TopK_TextBoxes", most_probable_TN_List_TopK_TextBoxes)