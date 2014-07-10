import numpy as np
import cv2
#from matplotlib import pyplot as plt

def train():
	img = np.zeros((600,600), np.uint8)
	cv2.line(img,(0,0),(0,600),255,2)
	cv2.line(img,(0,0),(600,0),255,2)
	cv2.line(img,(0,600),(600,600),255,2)
	cv2.line(img,(0,36),(168,36),255,2)
	cv2.line(img,(0,564),(168,564),255,2)
	cv2.line(img,(0,204),(228,204),255,2)
	cv2.line(img,(0,396),(228,396),255,2)
	cv2.line(img,(228,204),(228,396),255,2)
	cv2.line(img,(336,0),(336,36),255,2)
	cv2.line(img,(336,600),(336,564),255,2)

	cv2.ellipse(img,(63, 300),(285,285), 0, -68, 68, 255,2)
	cv2.ellipse(img,(228, 300),(72,72), 0, -90, 90, 255,2)
	cv2.circle(img, (63, 300), 7, 255, 1)
	for x in range(0, 600, 60):
		cv2.line(img, (x, 0), (x, 600), 60, 1)
	for y in range(0, 600, 60):
		cv2.line(img, (0, y), (600, y), 60, 1)
	return img
	


if __name__ == '__main__':
	while(True):
		cv2.imshow('detected', train())
		k = cv2.waitKey(50) & 0xFF
		if k == 27: break
	cv2.destroyAllWindows()