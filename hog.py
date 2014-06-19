import numpy as np
import cv2
from train import train

def draw_circle(img, rects):
	for x, y, w, h in rects:
		pad_h = int(0.05*h)
		cv2.ellipse(img, (x+w/2,y+h-pad_h), (w/3, w/5), 0, 0, 360, (255,255,255), 2)
		cv2.circle(img, (x+w/2,y+h-pad_h), 3, (255,255,255), -1)

def draw_detections(img, rects, thickness = 2, weight = None):
	for i, (x, y, w, h) in enumerate(rects):
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
		#cv2.putText(img,'(%s,%s)'%(x,y+h),(x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
		
key1 = np.loadtxt(open('data1.csv', 'rb'), delimiter=',')
key2 = np.loadtxt(open('data2.csv', 'rb'), delimiter=',')
key3 = np.loadtxt(open('data3.csv', 'rb'), delimiter=',')
key4 = np.loadtxt(open('data4.csv', 'rb'), delimiter=',')
keys = np.hstack((key1, key2, key3, key4)).reshape(-1, 4, 2).astype(np.float32)
pts = np.float32([[228,228],[228,372], [0,36], [0,396]])
tpl = train()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('CourtMapping.avi',fourcc, 20.0, (1920,720))

cv2.namedWindow('frame')

cap = cv2.VideoCapture('nba4_clip.avi')
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

num = 1
ret, frame = cap.read()
tpl_h,tpl_w = tpl.shape
frame_h, frame_w, _ = frame.shape
M = cv2.getPerspectiveTransform(keys[0], pts)
N = cv2.getPerspectiveTransform( pts, keys[0])
found, w = hog.detectMultiScale(frame, 
	winStride=(8,8), padding=(32,32), 
	scale=1.05, finalThreshold=-0.5, hitThreshold=-0.5)
arg = np.argsort(w, axis=0)
windows = found[arg[-1][0]]
draw_detections(frame, [windows])
while(cap.isOpened()):
	num += 1
	ret, frame = cap.read()
	M = cv2.getPerspectiveTransform(keys[num-1], pts)
	N = cv2.getPerspectiveTransform( pts, keys[num-1])
	blank = np.zeros_like(frame)
	found, w = hog.detectMultiScale(frame, 
			winStride=(8,8), padding=(32,32), 
			scale=1.05, finalThreshold=1.0, hitThreshold=-0.4)
	found = filter(lambda x:x[3]<300, found)
	#dists = np.apply_along_axis(np.linalg.norm, 1, found-windows)
	#idx = np.argmin(dists)
	#if dists[idx] < 100:
	#	windows = found[idx]
	draw_detections(frame, found)
	draw_circle(frame, found)
	draw_circle(blank, found)
	tplRot = cv2.warpPerspective(tpl, N, (frame_w, frame_h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	frame = cv2.addWeighted(tplRot2, 0.2, frame, 0.8, 0)
	blank_map = cv2.warpPerspective(blank, M, (tpl_w,tpl_h))
	tplC = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
	dst = cv2.addWeighted(blank_map, 0.5, tplC, 0.5, 0)
	wrap = cv2.copyMakeBorder(dst, 60, 60, 20, 20, cv2.BORDER_CONSTANT, value=0)
	mix = np.hstack((frame, wrap))
	cv2.imshow('frame', mix)
	out.write(mix)
	if cv2.waitKey(50) & 0xFF == 27:
		break
	#import pdb; pdb.set_trace()

cap.release()
out.release()
cv2.destroyAllWindows()