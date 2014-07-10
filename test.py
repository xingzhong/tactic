import numpy as np
import cv2


if __name__ == '__main__':
	cap = cv2.VideoCapture('nba4_clip.avi')
	cv2.namedWindow('frame')
	while (cap.isOpened()):
		ret, frame = cap.read()
		cv2.imshow('frame', frame)
		if cv2.waitKey(50) & 0xFF == 27:
			break
cap.release()
cv2.destroyAllWindows()