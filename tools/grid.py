import numpy as np
import cv2
from court import template, POI, HOOPS

cap = cv2.VideoCapture('../nba4_clip.avi')
tpl = template()

while(cap.isOpened()):
	ret, frame = cap.read()
	cv2.imshow('raw', frame)
	cv2.imshow('court', tpl)
	k = cv2.waitKey(50) & 0xFF
	if k == 27: break