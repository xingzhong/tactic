import numpy as np
import cv2
from simulate import background
font = cv2.FONT_HERSHEY_SIMPLEX
def p2c(r, t):
	return (int(r*np.cos(t))+250,
			 int(r*np.sin(t))+47 )

def init():
	r1, t1 = 300, .4*np.pi
	r2, t2 = r1 - 40 - 1*np.random.randn(), t1 + np.pi/36.0 * np.random.randn()
	r3, t3 = np.random.randint(r2, high=r1), t1-np.pi/36.0
	return (r1,t1,r2,t2,r3,t3)

def update(r1,t1,r2,t2,r3,t3, delta=0.15):
	r2 += delta * (r1-r2)
	t2 += delta * (t1-t2)
	r1 += delta * (r3 - 10 - r1)
	t1 += delta * (t3 - np.pi/36.0-t1)
	r3 += 0
	t3 += 0
	return (r1,t1,r2,t2,r3,t3)

r1,t1,r2,t2,r3,t3 = init()
for i in range(15):
	bg = background()
	print r1,t1,r2,t2,r3,t3
	cv2.putText(bg, "1", p2c(r1,t1), font, 0.5, (200,22,0), 2)
	cv2.putText(bg, "2", p2c(r2,t2), font, 0.5, (0,22,200), 2)
	cv2.putText(bg, "3", p2c(r3,t3), font, 0.5, (200,22,0), 2)
	cv2.imshow('img',bg)
	r1,t1,r2,t2,r3,t3 = update(r1,t1,r2,t2,r3,t3)
	if cv2.waitKey(500) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()