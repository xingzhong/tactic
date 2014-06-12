import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"X264")
out = cv2.VideoWriter('detected.mpg',fourcc, 20, (640,360))

for i in range(1, 178):
    f = 'dst/%s.png'%i
    img = cv2.imread(f, 0)
    out.write(img)
    cv2.imshow('frame',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()