import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt
from scipy.optimize import fmin
from icp import icp
def remove(img):
    img[540:640, 800:1210] = 0
    return img

def func(x, tpl, blank):
	h,w = blank.shape[:2:]
	x = x.reshape(3,3)
	tplRot = cv2.warpPerspective(tpl,x,(w,h))
	return -np.sum(np.logical_and(blank, tplRot))

cap = cv2.VideoCapture('nba4_clip.avi')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('cali.avi',fourcc, 20.0, (640,480))
#cimg = cv2.imread("fp60raw.png", 0)
tpl = train()
#tpl = cv2.imread("nba_court_small.png", 0)

pts1 = np.float32([[0,0],[228,228],[228,372], [0,396]])
pts2 = np.float32([[437,283],[707,403],[627,482], [132,457]])
M = cv2.getPerspectiveTransform(pts1,pts2)
print M

matched = None
kernel = np.ones((5,5),np.uint8)
tpl = cv2.dilate(tpl,kernel,iterations = 4)
tplKeyIdx = np.array([[0,0], [0, 36], [0, 205], [0,396], [0, 565], [0, 599],
						[228, 205], [228,228],[228,372], [228, 396], [300, 300],
						[350, 300], [0, 120], [0, 480], [170, 36], [171,565],
						[276, 111], [280, 483], [160, 205], [128, 396],
						[260, 0], [290, 599], [110, 0], [110, 599]], dtype=int)
tplKey = np.zeros_like(tpl)
#import pdb; pdb.set_trace()
tplKey[tplKeyIdx[:,1], tplKeyIdx[:,0]] = 255
while(cap.isOpened()):
	ret, frame = cap.read()
	blank = np.zeros(frame.shape[:2]).astype(np.uint8)
	edges = cv2.Canny(frame,10,150)
	edges = remove(edges)
	img, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-80:]
	cv2.drawContours(blank, contours, -1, 128, 1)
	#cv2.imshow('detected', blank)
	#out.write(dst)
	#k = cv2.waitKey(50000) & 0xFF
	#if k == 27: break
	#res = fmin(lambda x: func(x, tpl, blank), M)
	h,w = blank.shape[:2:]
	#M = res.reshape(3,3)
	
	tplRot = cv2.warpPerspective(tpl, M, (w,h))
	tplKeyRot = cv2.warpPerspective(tplKey, M, (w,h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	
	dst = cv2.addWeighted(tplRot2, 0.4, frame,0.6, 0)
	score = np.sum(np.logical_and(blank, tplRot))
	cv2.putText(dst, str(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	
	pos = (0,0,0)
	if matched is not None:
		#dilate = cv2.dilate(matched,kernel,iterations = 2)
		#mask = 255*np.logical_and(blank, dilate).astype(np.uint8)
		#cv2.imshow('detected', mask)
		#import pdb; pdb.set_trace()
		src = np.transpose(np.nonzero(matched)).T
		tgt = np.transpose(np.nonzero(blank)).T
		tr, pos = icp(src, tgt, init_pose=pos, no_iterations=30)
		M = np.dot(tr, M)
		
		cv2.imshow('detected', cv2.addWeighted(tplRot, 0.6, blank, 0.4, 0))
		#cv2.imshow('detected', tplKeyRot)
		k = cv2.waitKey(50) & 0xFF
		if k == 27: break
	#matched = tplKeyRot
	matched = 255*np.logical_and(blank, tplRot).astype(np.uint8)
    
cap.release()
#out.release()
cv2.destroyAllWindows()
