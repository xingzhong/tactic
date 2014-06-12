import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

def remove(img):
    img[540:640, 800:1210] = 0
    return img

orientations = 9
sy, sx = (1280, 720)
cx, cy = (8, 8)
bx, by = (2, 2)
n_cellsx = int(np.floor(sx // cx))
n_cellsy = int(np.floor(sy // cy))
n_blocksx = (n_cellsx - bx) + 1
n_blocksy = (n_cellsy - by) + 1
w, h = 28, 12

cap = cv2.VideoCapture('nba4_clip.avi')
cv2.namedWindow('slice')
cv2.createTrackbar('x','slice',0,n_cellsy-w,lambda x:x)
cv2.createTrackbar('y','slice',0,n_cellsx-h,lambda x:x)

truth = [(131,29), (121,23), (114, 28), (99, 25), (70, 29), (42, 48), 
            (59, 54), (116, 41)]

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = remove(gray)
    g = gray.copy()
    pyramidGray = [g]
    #for i in xrange(2):
    #    g = cv2.pyrDown(g)
    #    pyramidGray.append(g)

    for pic in pyramidGray:
        fd, hog_image = hog(pic, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True, normalise=True)
        feature = fd.reshape(n_blocksy, n_blocksx, by*bx*orientations)
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                xx = cx * x
                yy = cy * y
                if (xx%10 == 0) and (yy%10 == 0):
                    #gray = cv2.rectangle(gray, (xx,yy), (xx+cx, yy+cy), 255, 2)
                    pass
                else:
                    #gray = cv2.rectangle(gray, (xx,yy), (xx+cx, yy+cy), 200, 1)
                    pass

        cv2.imshow('image', gray)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        #while(True):
        #    x = cv2.getTrackbarPos('x','slice')
        #    y = cv2.getTrackbarPos('y','slice')
        #    xx = cx * x
        #    yy = cy * y
        #    cv2.imshow('slice', gray[yy:yy+cy*w, xx:xx+cx*h])
        #    if cv2.waitKey(50) & 0xFF == ord('q'):
        #        break
        trainData = np.empty(w*h*by*bx*orientations, dtype=np.float32)
        responses = np.empty(1, dtype=np.float32)
        for x in range(n_cellsy-h):
            for y in range(n_cellsx-w):
                if (x, y) in truth:
                    print x, y, (x, y) in truth
                    xx = cx * x
                    yy = cy * y
                    #import pdb; pdb.set_trace()
                    f = feature[ x:x+h, y:y+w].ravel().astype(np.float32)
                    trainData = np.vstack((trainData, f))
                    responses = np.vstack((responses, [1.0]))
                    cv2.imshow('feature', hog_image[yy:yy+cy*w, xx:xx+cx*h])
                    cv2.imshow('slice', gray[yy:yy+cy*w, xx:xx+cx*h])
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
        
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                        svm_type = cv2.SVM_C_SVC,
                        C=2.67, gamma=5.383 )

    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

    svm = cv2.SVM()
    import pdb; pdb.set_trace()
    svm.train(trainData, responses, params=svm_params)
    import pdb; pdb.set_trace()

cap.release()
cv2.destroyAllWindows()
