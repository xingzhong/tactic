import numpy as np
import cv2
from matplotlib import pyplot as plt

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
	return img
	


if __name__ == '__main__':
	cv2.imshow('detected', train())
	cv2.waitKey(0)
	cv2.destroyAllWindows()