import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('nba4.avi')
fgbg = cv2.BackgroundSubtractorMOG()
init = True
height = 256
width = 128

while(cap.isOpened()):
    ret, frame = cap.read()
    if init:
    	X, Y, _ = frame.shape
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	edges = cv2.Canny(gray, 40,200, apertureSize=3)
    	laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    	ret, thresh = cv2.threshold(gray, 100, 255,0)
    	contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    	circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,100)
    	#import pdb; pdb.set_trace()
    	#cv2.drawContours(frame, contours, -1, (0,255,0), 2)
    	for (x,y,radius) in np.squeeze(circles):
		    cv2.circle(gray,(x,y),radius, (0,255,255),2)

    	plt.imshow(gray)
    	plt.show()
    	init = False
    	break
    
    #h, w = gray.shape
    #cv2.rectangle(gray,(w/2-100,h/2-200),(w/2+100,h/2+200),(0,255,0),3)
    #edges = cv2.Canny(gray,100,200)

    #fgmask = fgbg.apply(gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()