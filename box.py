import numpy as np
import cv2
from train import train
from sklearn.neighbors import NearestNeighbors

COLORS = np.random.random_integers(0, high=255, size=(100, 3))

def foot(rect):
	x, y, w, h = rect
	pad_w, pad_h = int(0.15*w), int(0.05*h)
	return (x+w/2,y+h-pad_h)

def draw_map(img, circles):
	r = 10
	for circle in circles[-5:]:
		r -= 2 
		for (i, (x, y)) in enumerate(circle):
			#import pdb; pdb.set_trace()
			cv2.circle(img, (int(x),int(y)), r, COLORS[i].tolist(), -1)


def draw_detections(img, rects, thickness = 1, weight = None):
	for rect in rects[-1:]:
		for (i, (x, y, w, h)) in enumerate(rect):
			# the HOG detector returns slightly larger rectangles than the real objects.
			# so we slightly shrink the rectangles to get a nicer output.
			#import pdb; pdb.set_trace()
			pad_w, pad_h = int(0.15*w), int(0.05*h)
			sample = img[y:y+h, x:x+w].copy()
			cv2.ellipse(img, (x+w/2,y+h-pad_h), (w/3, w/5), 0, 0, 360, (250,0,0), 2)
			cv2.circle(img, (x+w/2,y+h-pad_h), 3, (250,0,0), -1)
			cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), COLORS[i].tolist(), thickness)
			#cv2.putText(img,'(%s,%s)'%(x,y+h),(x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
		
key1 = np.loadtxt(open('data1.csv', 'rb'), delimiter=',')
key2 = np.loadtxt(open('data2.csv', 'rb'), delimiter=',')
key3 = np.loadtxt(open('data3.csv', 'rb'), delimiter=',')
key4 = np.loadtxt(open('data4.csv', 'rb'), delimiter=',')
keys = np.hstack((key1, key2, key3, key4)).reshape(-1, 4, 2).astype(np.float32)
pts = np.float32([[228,228],[228,372], [0,36], [0,396]])
tpl = train()
Found1 = np.loadtxt(open('player1.csv', 'rb'), delimiter=',')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('CourtMapping.avi',fourcc, 20.0, (1920,720))

cv2.namedWindow('frame')

cap = cv2.VideoCapture('nba4_clip.avi')

num = 1
ret, frame = cap.read()
tpl_h,tpl_w = tpl.shape
frame_h, frame_w, _ = frame.shape
M = cv2.getPerspectiveTransform(keys[0], pts)
N = cv2.getPerspectiveTransform( pts, keys[0])

#roi = cv2.perspectiveTransform(found_map_1.reshape(-1,1,2), M).reshape(-1,2)

Found = []
Found_map = []
while(cap.isOpened()):
	num += 1
	ret, frame = cap.read()
	M = cv2.getPerspectiveTransform(keys[num-1], pts)
	N = cv2.getPerspectiveTransform( pts, keys[num-1])
	blank = np.zeros_like(frame)
	tplC = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)

	
	#foundFoot = map(foot, found)
	#found_map_1 = np.float32(foundFoot)[:, :2]
	#found_map = cv2.perspectiveTransform(found_map_1.reshape(-1,1,2), M).reshape(-1,2)
	#nbrs = NearestNeighbors(n_neighbors=1).fit(found_map)
	#distances, indices = nbrs.kneighbors(roi)
	#roi = found_map[indices].reshape(-1,2)

	
	#Found.append(np.array(found)[indices].reshape(-1,4))
	#Found_map.append(roi)

	#draw_map(tplC, Found_map)
	draw_detections(frame, Found)
	
	tplRot = cv2.warpPerspective(tpl, N, (frame_w, frame_h))
	tplRot2 = cv2.cvtColor(tplRot, cv2.COLOR_GRAY2BGR)
	frame = cv2.addWeighted(tplRot2, 0.2, frame, 0.8, 0)
	#blank_map = cv2.warpPerspective(blank, M, (tpl_w,tpl_h))

	#dst = cv2.addWeighted(blank_map, 0.5, tplC, 0.5, 0)
	wrap = cv2.copyMakeBorder(tplC, 60, 60, 20, 20, cv2.BORDER_CONSTANT, value=0)
	mix = np.hstack((frame, wrap))

	cv2.imshow('frame', mix)
	#out.write(mix)
	if cv2.waitKey(10000) & 0xFF == 27:
		break
	import pdb; pdb.set_trace()

cap.release()
#out.release()
cv2.destroyAllWindows()