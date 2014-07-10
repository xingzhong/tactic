import numpy as np
import cv2

POI = [(75, 68), (75, 113), (75, 296), (75, 323), (75, 496), (75, 520), 
		(75, 712), (75, 756), (328, 297), (328, 326), (328, 496), 
		(328, 520), (435, 68), (435, 756), (638, 68), (638, 323),
		(638, 498), (638, 756), (839, 68), (839, 756), (945, 297),
		(945, 324), (945, 496), (945, 518), (1201, 68), (1201, 113), 
		(1201, 296), (1201, 323), (1201, 496), (1201, 520), (1201, 712), (1201, 756),
		(638, 410), (490, 410), (790, 410)]

HOOPS = [(148, 410), (1126, 410)]
def template():
	img = cv2.imread("nba_court.png", cv2.IMREAD_COLOR)
	height, width = img.shape[:2]
	res = cv2.resize(img,(int(.8*width), int(.8*height)), interpolation = cv2.INTER_CUBIC)
	for (x, y) in POI:
		cv2.circle(res, (x, y), 8, (0,0,255), 2)
	for (x, y) in HOOPS:
		cv2.circle(res, (x, y), 5, (255,0,0), -1)
	return res




if __name__ == '__main__':
	while(True):
		cv2.imshow('template', template())
		k = cv2.waitKey(50) & 0xFF
		if k == 27: break
	cv2.destroyAllWindows()