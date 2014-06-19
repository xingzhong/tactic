import numpy as np
import cv2
from matplotlib import pyplot as plt


x1,y1 = -1, -1
track_window = [-1,-1,-1,-1]
def click(event,x,y,flags,params):
    global x1,y1, roi, track_window
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(params, (x1,y1), (x,y), (0,0,255), 1)
        roi = params[min(y,y1):max(y,y1), min(x,x1):max(x,x1)]
        track_window[0] =  min(x,x1)
        track_window[1] =  min(y,y1)
        track_window[2] =  max(x,x1) - min(x,x1)
        track_window[3] =  max(y,y1) - min(y,y1)

def feature(img):
    hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0, 1], None,[128, 128],[0, 128, 0, 128])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    #plt.imshow(roi_hist, cmap='gray')
    #plt.show()
    return roi_hist

cv2.namedWindow('frame')        
cap = cv2.VideoCapture('long.avi')
roi = None
# take first frame of the video
ret,frame = cap.read()
cv2.setMouseCallback('frame',click, frame)
while(True):
    cv2.imshow('frame', frame)
    if cv2.waitKey(50) & 0xFF == 27:
        break

roi_hist = feature(roi)
#import pdb; pdb.set_trace()


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 3 )

while(cap.isOpened()):
    ret ,frame = cap.read()
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,128,0,128],1)
        
        #import pdb; pdb.set_trace()
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(bp, tuple(track_window), term_crit)
        cv2.normalize(bp,bp,0,255,cv2.NORM_MINMAX)
        x,y,w,h = track_window
        #roi = frame[y:y+h, x:x+w]
        #roi_hist = feature(roi)
        #x, y = maxLoc
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        bp = cv2.rectangle(bp, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('video',img2)
        cv2.imshow('bp',bp)
        cv2.imshow('roi',roi)

        k = cv2.waitKey(40) & 0xff
        if k == 27:
            break
        #import pdb; pdb.set_trace()
        #else:
        #    cv2.imwrite(chr(k)+".jpg",img2)

cv2.destroyAllWindows()
cap.release()