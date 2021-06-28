import cv2
import os
import numpy as np
from mtcnn import MTCNN
from cv2 import CascadeClassifier
from cv2 import rectangle
from imutils.object_detection import non_max_suppression
import time
import pytesseract

path='Train/'


files=os.listdir(path)
def face_detector():
    index = 0
    detector = MTCNN()
    for f in files:
        img = cv2.imread('Train/' + str(index) + '.jpg')

        cropped_image = img[5:225, 20:360]
        cv2.imshow("img", cropped_image)
        cv2.waitKey(0)
        classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
        boxes = classifier.detectMultiScale(cropped_image)
        for box in boxes:
            x, y, w, h = box
            x2, y2 = x + w, y + h
            face = cropped_image[y: y2, x: x2]
            cv2.imshow('face', face)
            cv2.waitKey(0)
            rectangle(cropped_image, (x, y), (x2, y2), (0, 0, 255), 1)
        # cv2.imshow('face',cropped_image)
        # cv2.waitKey(0)
        index += 1


def text_detector():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    net=cv2.dnn.readNet("frozen_east_text_detection.pb")
    index=0
    for f in files:
        img = cv2.imread(('Train/'+str(index)+'.jpg'))
        (H,W)=img.shape[:2]
        newW,newH=(224,352)
        rW=W/float(newW)
        rH=H/float(newH)
        cropped_image = cv2.resize(img,(newW,newH))
        (H,W)=cropped_image.shape[:2]
        layerNames=[
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        blob = cv2.dnn.blobFromImage(cropped_image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(0, numCols):
                if scoresData[x] < 0.5:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            box=img[startY:endY,startX:endX]
            cv2.imshow("Text", box)
            text=pytesseract.image_to_string(box,lang="ben")
            print(text)
            cv2.waitKey(0)
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output image
        #cv2.imshow("Text Detection", img)
        #cv2.waitKey(0)
        index+=1

face_detector()
#text_detector()

