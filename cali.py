import cv2
import numpy as np
from train import train
from matplotlib import pyplot as plt

cimg = cv2.imread("fp60raw.png", 0)
blank = np.zeros_like(cimg)
edges = cv2.Canny(cimg,0,150)
img, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x:cv2.arcLength(x, False))[-50:]
cv2.drawContours(blank, contours, -1, 128, 1)
tpl = train()
#tpl = cv2.imread("nba_court_small.png", 0)

h,w = blank.shape[:2:]
pts1 = np.float32([[0,36],[228,228],[228,372], [336,0]])
pts2 = np.float32([[496,280],[822,405],[735,490], [1147,311]])

M = cv2.getPerspectiveTransform(pts1,pts2)

tpl = cv2.warpPerspective(tpl,M,(w,h))
dst = cv2.addWeighted(cimg, 0.5, tpl, 0.5, 0)

cv2.imshow('detected', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
