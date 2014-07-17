import cv2
import numpy as np


def warpPerspective(src, M, size):
	dst = np.zeros((1.1*size[1], 1.1*size[0],3), dtype=np.uint8)
	src_x, src_y = src.shape[:2]
	msrc = np.vstack(np.meshgrid(range(src_x), range(src_y))).reshape(2, -1)
	msrc = np.vstack((msrc, np.ones((1, src_x*src_y))))
	mdst = np.dot(M, msrc)[:2, :].astype(int)
	import pdb; pdb.set_trace()
	dst[mdst[1,:],mdst[0,:], :] = src.reshape(-1, 3)
	return dst

if __name__ == '__main__':
	M = np.array([[2.54, -1.52, 290.5], [0.134, 0.232, 248.1], [0,0,1]])
	size = (1280, 720)
	src = cv2.imread("../court/Oklahoma_City_Thunder_court.png")
	dst = warpPerspective(src, M, size)
	while(True):
		cv2.imshow('src', src)
		cv2.imshow('dst', dst)
		k = cv2.waitKey(500) & 0xFF
		if k == 27: break
	cv2.destroyAllWindows()
	