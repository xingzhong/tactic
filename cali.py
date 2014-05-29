import cv2
import numpy as np

cimg = cv2.imread("fp60raw.png", 0)
circles = cv2.HoughCircles(cimg, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, 
			param1=50, param2=30, minRadius=300, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:20]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0,255,0), 2)
    cv2.circle(cimg, (i[0], i[1]), 20, (0,0,255), 3)

cv2.imshow('detected', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
