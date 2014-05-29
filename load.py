import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('nba4.avi')
fgbg = cv2.BackgroundSubtractorMOG()
init = True
height = 256
width = 128

num = 0
while(cap.isOpened()):
    num += 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40,200, apertureSize=3)
    cv2.imshow('frame', edges)
    if num == 60:
        cv2.imwrite("fp60raw.png", gray)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()
