import numpy as np
import cv2
from train import train
from sklearn.mixture.gmm import log_multivariate_normal_density
import csv

Color = {'rb' : (255, 65, 105), 'tq' : (208, 64, 224), 
		'sb': (96, 244, 164), 'pl': (221, 221, 160),
		'ls': (122, 255, 160)}

def polar(xy, zero=(63, 300)):
	x, y = xy
	x0, y0 = zero
	x = x-x0
	y = y-y0
	r = np.sqrt(x**2+y**2)
	theta = np.arctan2(y, x)
	return np.array([r, theta])

def Degree(ps):
	p1, p2, p3, p4 = map(np.array, ps)
	x = np.array([p4-p1, p2-p1])
	n  = np.linalg.norm(x, axis=1)
	nx = x[0, :]/n[0]
	ny = x[1, :]/n[1]
	#import pdb; pdb.set_trace()
	theta = 57.3* np.arccos(np.dot(nx, ny))
	return theta
	
def Split(ps):
	p1, p2, p3, p4 = map(np.array, ps)
	return np.linalg.norm(p1-p2)

def DScore(ps):
	p1, p2, p3, p4 = map(polar, ps)
	means = np.array([50, 0]).reshape(1,2)
	covars = np.array([5, .1]).reshape(1,2)
	x = np.array([p1-p3, p2-p4])
	s = log_multivariate_normal_density(x, means, covars)
	return np.sum(s)

def path(file):
	p = np.genfromtxt(file, delimiter=',', dtype=int)
	xp, fp1, fp2 = p[:,0], p[:,1], p[:,2]
	t = range(min(xp), max(xp))
	x = np.interp(t, xp, fp1)
	y = np.interp(t, xp, fp2)
	return  np.vstack((t, x, y)).T

if __name__ == '__main__':
	font = cv2.FONT_HERSHEY_SIMPLEX
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('pp.avi',fourcc, 20.0, (600,600))
	p1 = path("player1.csv")
	p2 = path("player2.csv")
	p3 = path("player3.csv")
	p4 = path("player4.csv")
	ball = path("ball.csv")
	f2d = []
	for (t, x1, y1, _, x2, y2, _, x3, y3, _, x4, y4, _, x5, y5) in np.hstack((p1,p2,p3,p4,ball)):
		bg = train()
		bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

		cv2.circle(bg, (int(x1), int(y1)), 10, Color['rb'], 2)
		cv2.circle(bg, (int(x2), int(y2)), 10, Color['tq'], 2)
		cv2.circle(bg, (int(x3), int(y3)), 10, Color['sb'], 2)
		cv2.circle(bg, (int(x4), int(y4)), 10, Color['pl'], 2)
		cv2.circle(bg, (int(x5), int(y5)), 5, Color['ls'], -1)
		pts = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
		defence = DScore(pts)
		degree = Degree(pts)
		split = Split(pts)

		cv2.line(bg, (63, 300), (int(x1), int(y1)), Color['rb'], 1)
		cv2.line(bg, (63, 300), (int(x2), int(y2)), Color['tq'], 1)
		cv2.line(bg, (63, 300), (int(x3), int(y3)), Color['sb'], 1)
		cv2.line(bg, (63, 300), (int(x4), int(y4)), Color['pl'], 1)

		cv2.line(bg, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
		#cv2.line(bg, (int(x1), int(y1)), (int(x3), int(y3)), (255,0,0), 1)
		cv2.line(bg, (int(x1), int(y1)), (int(x4), int(y4)), (255,255,255), 2)
		#cv2.line(bg, (int(x2), int(y2)), (int(x4), int(y4)), (255,255,255), 1)
		#cv2.line(bg, (int(x2), int(y2)), (int(x4), int(y4)), (255,0,0), 1)
		#cv2.line(bg, (int(x3), int(y3)), (int(x4), int(y4)), (255,0,0), 1)

		cv2.putText(bg, "# %s"%t, (450,450), font, 1, (255,255,255), 1)
		cv2.putText(bg, "D %s"%int(defence), (450,490), font, 1, (255,255,255), 1)
		cv2.putText(bg, "T %s"%int(degree), (450,530), font, 1, (255,255,255), 1)
		cv2.putText(bg, "S %s"%int(split), (450,570), font, 1, (255,255,255), 1)
		f2d.append((int(defence), int(degree), int(split)))

		cv2.imshow('img',bg)
		out.write(bg)
		if cv2.waitKey(50) & 0xFF == 27:
			break
		if t%10 == 3:
			while(1):
				cv2.imshow('img',bg)
				k = cv2.waitKey(100) & 0xFF 
				if k == ord('c'):
					break
	with open('features.csv', 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(f2d)
	out.release()
	cv2.destroyAllWindows()