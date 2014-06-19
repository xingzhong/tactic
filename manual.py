import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt
from scipy.optimize import fmin, fmin_powell

def subpic(img, key, margin):
	return img[(key[1]-margin[1]):(key[1]+margin[1]),
				(key[0]-margin[0]):(key[0]+margin[0])]

def seekMax(img, loc, mar):
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
	#print max_val
	if max_val > 0.2:
		loc = max_loc[0]+mar[0], max_loc[1]+mar[1]
	return loc

def maskROI(img, loc, search):
	new = np.zeros_like(img)
	new[(loc[1]-search[1]):(loc[1]+search[1]),
				(loc[0]-search[0]):(loc[0]+search[0])] = subpic(img, loc, search)
	return new 

def RoiMap(keyPoints):
	for i, img in enumerate(keyPoints):
		plt.subplot(2, 1+len(keyPoints)/2, i+1)
		plt.imshow(keyPoints[i], cmap='gray')
	plt.show()

def func(x, tpl, blank):
	h,w = blank.shape[:2:]
	Tr = np.array([[np.cos(x[2]),-np.sin(x[2]),x[0]],
                   [np.sin(x[2]), np.cos(x[2]),x[1]],
                   [0,0,1]])
	tplRot = cv2.warpPerspective(tpl,Tr,(w,h))
	return -np.sum(np.logical_and(blank, tplRot))

cap = cv2.VideoCapture('nba4_clip.avi')
tpl = train()

pts1 = np.float32([[0,0],[228,228],[228,372], [0,396]])
pts2 = np.float32([[437,283],[707,403],[627,482], [132,457]])
M = cv2.getPerspectiveTransform(pts1,pts2)

N = np.eye(3)
x = [0,0,0]
init = True

key = [(707, 403), (627, 482), (437,283), (87,480), (266, 557)]
locs = [(707, 403), (627, 482), (437,283), (87,480), (266, 557)]
mar = (24, 24)
search = (48, 48)

cv2.namedWindow('detected')
cv2.createTrackbar('x0','detected',0,200, lambda x:x)
cv2.createTrackbar('x1','detected',0,200, lambda x:x)
cv2.createTrackbar('x2','detected',0,200, lambda x:x)
frameNum = 0
data = []
while(cap.isOpened()):
	ret, frame = cap.read()
	frameNum += 1
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blank = np.zeros(frame.shape[:2]).astype(np.uint8)
	edges = cv2.Canny(frame,10,150)
	img, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-15:]
	cv2.drawContours(blank, contours, -1, 128, 1)
	if init:
		h,w = blank.shape[:2:]
		tplRot = cv2.warpPerspective(tpl, M, (w,h))
		oldx = [0.0,0.0,0.0]
		init = False
	else:
		while(1):
			x[0] = oldx[0] + (cv2.getTrackbarPos('x0','detected')-100)
			x[1] = oldx[1] + (cv2.getTrackbarPos('x1','detected')-100)/18.0
			x[2] = oldx[2] + (cv2.getTrackbarPos('x2','detected')-100)/180.0
			N = np.array([[np.cos(x[2]),-np.sin(x[2]),x[0]],
                   [np.sin(x[2]), np.cos(x[2]),x[1]],
                   [0,0,1]])
			newTpl = cv2.warpPerspective(tplRot, N, (w,h))
			tplRot2 = cv2.cvtColor(newTpl, cv2.COLOR_GRAY2BGR)
			score = np.sum(np.logical_and(blank, newTpl))
			dst = cv2.addWeighted(tplRot2, 0.4, frame, 0.6, 0)
			cv2.putText(dst,'%s'%frameNum,(10,500), 
						cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
			cv2.imshow('detected', dst)

			k = cv2.waitKey(50) & 0xFF
			if k == 27: 
				oldx[:] = x[:]
				tplRot = newTpl.copy()
				data.append(x[:])
				break
	
	print data
import pdb; pdb.set_trace()

cap.release()
#out.release()
cv2.destroyAllWindows()