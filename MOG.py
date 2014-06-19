import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt


def draw_detections(img, rects, thickness = 2, weight = None):
	for i, (x, y, w, h) in enumerate(rects):
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
		cv2.putText(img,'(%s,%s)'%(x,y+h),(x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)

key1 = np.loadtxt(open('data1.csv', 'rb'), delimiter=',')
key2 = np.loadtxt(open('data2.csv', 'rb'), delimiter=',')
key3 = np.loadtxt(open('data3.csv', 'rb'), delimiter=',')
key4 = np.loadtxt(open('data4.csv', 'rb'), delimiter=',')
keys = np.hstack((key1, key2, key3, key4)).reshape(-1, 4, 2).astype(np.float32)
pts = np.float32([[228,228],[228,372], [0,36], [0,396]])
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('CourtMapping.avi',fourcc, 20.0, (1280,720))
cap = cv2.VideoCapture('nba4_clip.avi')
tpl = train()

frameNum = 0
next = True
#cv2.namedWindow('detected')
ret, frame = cap.read()
frameNum += 1
h,w,_ = frame.shape
M = cv2.getPerspectiveTransform(pts, keys[frameNum-1])
bg = frame.copy()
avg = np.zeros_like(frame, dtype=np.float32)
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setBackgroundRatio(1.0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

try:
	while(cap.isOpened()):
		ret, frame = cap.read()
		frameNum += 1
		h,w,_ = frame.shape
		#cv2.putText(frame,'%s'%frameNum,(10,500), 
		#					cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
		N = cv2.getPerspectiveTransform(keys[frameNum-2], keys[frameNum-1])
		M = np.dot(N, M)
		bg = cv2.warpPerspective(frame, N, (w,h))

		tplRot = cv2.warpPerspective(tpl, M, (w,h))
		tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
		dst = cv2.addWeighted(tplRot2, 0.2, frame, 0.8, 0)
		fgmask = fgbg.apply(frame)
		#found, w = hog.detectMultiScale(fgmask, 
		#		winStride=(8,8), padding=(32,32), 
		#		scale=1.05, finalThreshold=2.0, hitThreshold=0)
		#draw_detections(frame, found)
		#cv2.imshow('detected', dst)

		ret, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)
		cv2.imshow('bg', thresh)
		#cv2.imshow('bg2', fgbg.getBackgroundImage())
		out.write(dst)
		k = cv2.waitKey(50) & 0xFF
		if k == 27: break
		#import pdb; pdb.set_trace()
except:
	pass
cap.release()
out.release()
cv2.destroyAllWindows()