import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('nba4.avi')
init = True
height = 256
width = 128

num = 0
while(cap.isOpened()):
    num += 1
    ret, frame = cap.read()
    blank = 255*np.ones_like(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40,200, apertureSize=3)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-30:]
    cv2.drawContours(blank, contours, -1, 80, 3)
    cv2.imshow('frame', blank)
    if num == 60:
        cv2.imwrite("fp60raw.png", gray)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
