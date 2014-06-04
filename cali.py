import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt
from scipy.optimize import fmin

def func(x, tpl, blank):
	h,w = blank.shape[:2:]
	x = x.reshape(3,3)
	tplRot = cv2.warpPerspective(tpl,x,(w,h))
	return -np.sum(np.logical_and(blank, tplRot))

cap = cv2.VideoCapture('nba4_clip.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cali.avi',fourcc, 20.0, (640,480))
#cimg = cv2.imread("fp60raw.png", 0)
tpl = train()
#tpl = cv2.imread("nba_court_small.png", 0)

pts1 = np.float32([[0,36],[228,228],[228,372], [336,0]])
pts2 = np.float32([[496,280],[822,405],[735,490], [1147,311]])
M = cv2.getPerspectiveTransform(pts1,pts2)
print M
#import pdb; pdb.set_trace()
while(cap.isOpened()):
	ret, frame = cap.read()
	blank = np.zeros(frame.shape[:2]).astype(np.uint8)
	edges = cv2.Canny(frame,0,150)
	img, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-50:]
	cv2.drawContours(blank, contours, -1, 128, 1)
	res = fmin(lambda x: func(x, tpl, blank), M)
	h,w = blank.shape[:2:]
	M = res.reshape(3,3)
	tplRot = cv2.warpPerspective(tpl, M, (w,h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	#import pdb; pdb.set_trace()
	dst = cv2.addWeighted(tplRot2, 0.2,frame,0.8,0)
	score = np.sum(np.logical_and(blank, tplRot))
	cv2.putText(dst, str(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	cv2.imshow('detected', dst)
	out.write(dst)
	k = cv2.waitKey(50) & 0xFF
	if k == 27: break
    
cap.release()
out.release()
cv2.destroyAllWindows()
