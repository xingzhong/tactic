import numpy as np
import cv2
#import cv2.cv as cv
from matplotlib import pyplot as plt

def callback(img):
	pass

def drawHist(img):
	#h = 255*np.ones((300,130, 3))
	#margin = 10
	#bins = np.arange(110).reshape(110,1)
	mix = img[:,:,0]*img[:,:,1]+img[:,:,2]
	hist_item = cv2.calcHist([mix],[0],None,[110],[0,110])
	cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
	hist=np.int32(np.around(hist_item))
	#pts = np.column_stack((bins,hist))
	#cv2.polylines(h,[pts],False, (255,255,255))
	#for (x, y) in pts:
	#	cv2.line(h, (x+margin,margin), (x+margin,y+margin), (0,0,0), 1)
	#h=np.flipud(h)
	#
	return hist.astype(float), h
	
def similarity(hist1, hist2):
	return np.sqrt( 
			1 - np.sum(np.sqrt(hist1*hist2))/np.sqrt(np.mean(hist1)*np.mean(hist2)*110*110))

cap = cv2.VideoCapture('nba4_clip.avi')
w, h = 110, 220
cv2.namedWindow('slice')
cv2.createTrackbar('x','slice',0,1280-w, lambda x: callback(x))
cv2.createTrackbar('y','slice',0,720-h,  lambda x: callback(x))
X, Y = 1037, 235
tpl = None
while(cap.isOpened()):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if tpl is None:
    	tpl, tpl_img = drawHist(hsv[Y:Y+h, X:X+w])
    #while(1):
    detect = 255*np.ones((1280-w, 720-h))
    for x in range(0, 1280-w, 5):
    	for y in range(0, 720-h, 5):
    		hist, hist_img = drawHist(hsv[y:y+h, x:x+w])
    		detect[x, y] = similarity(hist, tpl)
    		#print x, y, detect[x, y]
    line = np.sort(detect.ravel())[10]
    rois = np.vstack(np.where(detect < line)).T
    for (x, y) in rois:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow('slice', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
    	break
    #import pdb; pdb.set_trace()
cap.release()
cv2.destroyAllWindows()