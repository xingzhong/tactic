import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt

pts = []
def click(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(param, (x,y),5,(255,0,0),3)
		print x, y
		pts.append((x, y))


key1 = np.loadtxt(open('data1.csv', 'rb'), delimiter=',')
key2 = np.loadtxt(open('data2.csv', 'rb'), delimiter=',')
key3 = np.loadtxt(open('data3.csv', 'rb'), delimiter=',')
key4 = np.loadtxt(open('data4.csv', 'rb'), delimiter=',')
keys = np.hstack((key1, key2, key3, key4)).reshape(-1, 4, 2).astype(np.float32)
pts = np.float32([[228,228],[228,372], [0,36], [0,396]])

cap = cv2.VideoCapture('nba4_clip.avi')
tpl = train()

frameNum = 0
next = True
cv2.namedWindow('detected')

font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('calibrationGrid.avi',fourcc, 20.0, (1280,720))

while(cap.isOpened()):
	ret, frame = cap.read()
	frameNum += 1
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blank = np.zeros(frame.shape[:2]).astype(np.uint8)
	h,w = blank.shape[:2:]
	edges = cv2.Canny(frame,10,150)
	img, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-15:]
	cv2.drawContours(blank, contours, -1, 128, 1)
	cv2.putText(frame,'%s'%frameNum,(10,500), 
						cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
	#cv2.circle(frame, tuple(key1[frameNum-1].astype(int)), 5,(0,0,255),3)
	#cv2.circle(frame, tuple(key2[frameNum-1].astype(int)), 5,(0,0,255),3)
	#cv2.circle(frame, tuple(key3[frameNum-1].astype(int)), 5,(0,0,255),3)
	#cv2.circle(frame, tuple(key4[frameNum-1].astype(int)), 5,(0,0,255),3)
	#cv2.setMouseCallback('detected',click, frame)
	M = cv2.getPerspectiveTransform(pts, keys[frameNum-1])
	#M_INV =  cv2.getPerspectiveTransform( keys[frameNum-1], pts)
	tplRot = cv2.warpPerspective(tpl, M, (w,h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	#frame_INV = cv2.warpPerspective(frame, M_INV, (w,h))
	dst = cv2.addWeighted(tplRot2, 0.2, frame, 0.8, 0)
	cv2.imshow('detected', dst)
	#out.write(dst)
	k = cv2.waitKey(50) & 0xFF
	if k == 27: break
	#import pdb; pdb.set_trace()
cap.release()
out.release()
cv2.destroyAllWindows()