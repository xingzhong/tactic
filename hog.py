import numpy as np
import cv2
from train import train
from matplotlib import pyplot as plt

def foot(rect):
	x, y, w, h = rect
	pad_w, pad_h = int(0.15*w), int(0.05*h)
	return (x+w/2,y+h-pad_h)

def draw_map(img, circle):
	colors = np.random.random_integers(0, high=255, size=(K, 3))
	for (x, y) in circle:
		cv2.circle(img, (int(x),int(y)), 6, colors[0].tolist(), -1)

def draw_center(img, centers):
	for (x, y) in centers:
		cv2.circle(img, (int(x),int(y)), 20, (0,0,255), 2)

def draw_detections(img, rects, thickness = 1, weight = None):
	for i, (x, y, w, h) in enumerate(rects):
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		sample = img[y:y+h, x:x+w].copy()
		cv2.ellipse(img, (x+w/2,y+h-pad_h), (w/3, w/5), 0, 0, 360, (250,0,0), 2)
		cv2.circle(img, (x+w/2,y+h-pad_h), 3, (250,0,0), -1)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 250, 0), thickness)
		#cv2.putText(img,'(%s,%s)'%(x,y+h),(x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
		
key1 = np.loadtxt(open('data1.csv', 'rb'), delimiter=',')
key2 = np.loadtxt(open('data2.csv', 'rb'), delimiter=',')
key3 = np.loadtxt(open('data3.csv', 'rb'), delimiter=',')
key4 = np.loadtxt(open('data4.csv', 'rb'), delimiter=',')
keys = np.hstack((key1, key2, key3, key4)).reshape(-1, 4, 2).astype(np.float32)
pts = np.float32([[228,228],[228,372], [0,36], [0,396]])
tpl = train()
K = 12

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('CourtMapping.avi',fourcc, 20.0, (1920,720))

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
	scale=1.05, finalThreshold=.9, hitThreshold=-0.5)
arg = np.argsort(w, axis=0)

found = found[arg].reshape(-1, 4)
found = filter(lambda x:x[3]<300, found)
windows = found[-5]
x, y, w, h = windows
roi = frame[y:y+h, x:x+w]
cv2.imshow('roi',  roi)
gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray,2,3,0.04)
roi[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',  roi)
while(cap.isOpened()):
	num += 1
	ret, frame = cap.read()
	M = cv2.getPerspectiveTransform(keys[num-1], pts)
	N = cv2.getPerspectiveTransform( pts, keys[num-1])
	blank = np.zeros_like(frame)
	tplC = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)

	found, w = hog.detectMultiScale(frame, 
			winStride=(8,8), padding=(32,32), 
			scale=1.05, finalThreshold=1, hitThreshold=-0.5)

	found = filter(lambda x:x[3]<300, found)
	#found = found[4:6]
	foundFoot = map(foot, found)
	found_map_1 = np.float32(foundFoot)[:, :2]
	found_map = cv2.perspectiveTransform(found_map_1.reshape(-1,1,2), M).reshape(-1,2)
	draw_map(tplC, found_map)
	
	#dists = np.apply_along_axis(np.linalg.norm, 1, found-windows)
	#idx = np.argmin(dists)
	#if dists[idx] < 100:
	#	windows = found[idx]
	draw_detections(frame, found)
	draw_detections(frame, [windows], thickness = 3)
	tplRot = cv2.warpPerspective(tpl, N, (frame_w, frame_h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	frame = cv2.addWeighted(tplRot2, 0.2, frame, 0.8, 0)
	#blank_map = cv2.warpPerspective(blank, M, (tpl_w,tpl_h))

	#dst = cv2.addWeighted(blank_map, 0.5, tplC, 0.5, 0)
	wrap = cv2.copyMakeBorder(tplC, 60, 60, 20, 20, cv2.BORDER_CONSTANT, value=0)
	mix = np.hstack((frame, wrap))

	cv2.imshow('frame', mix)
	#out.write(mix)
	if cv2.waitKey(50) & 0xFF == 27:
		break
	import pdb; pdb.set_trace()

cap.release()
#out.release()
cv2.destroyAllWindows()