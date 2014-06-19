import numpy as np
import cv2
from matplotlib import pyplot as plt

x1,y1 = -1, -1

def click(event,x,y,flags,img):
    global x1,y1
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img, (x1,y1), (x,y), (0,0,255), 1)

cap = cv2.VideoCapture('nba4_clip.avi')
num = 0
cv2.namedWindow('frame')
while(cap.isOpened()):
    num += 1
    ret, frame = cap.read()
    cv2.setMouseCallback('frame',click, frame)
    while(True):
        cv2.imshow('frame', frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
