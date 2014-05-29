import numpy as np
import cv2
from matplotlib import pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX



def background():
	img = 255*np.ones((470,500,3), np.uint8)
	cv2.line(img,(30,0),(30,140),(0,0,0),2)
	cv2.line(img,(170,0),(170,190),(0,0,0),2)
	cv2.line(img,(190,0),(190,190),(0,0,0),2)
	cv2.line(img,(310,0),(310,190),(0,0,0),2)
	cv2.line(img,(330,0),(330,190),(0,0,0),2)
	cv2.line(img,(470,0),(470,140),(0,0,0),2)

	cv2.line(img,(220,40),(280,40),(0,0,0),2)
	cv2.line(img,(170,190),(330,190),(0,0,0),2)

	cv2.circle(img,(250,47),7,(0,0,0),2)
	cv2.ellipse(img,(250,47),(40,40), 0, 0, 180, (0,0,0),2)
	cv2.ellipse(img,(250,47),(238,238),0, 22, 158, (0,0,0),2)
	cv2.circle(img,(250,190),60,(0,0,0),2)
	return img

def player(img, pos):
	for time, player, team, x, y in pos:
		if team == 1:
			color = (200,22,0)
			cv2.putText(img, str(player), 
					(x, y), font, 0.5, color, 2)
		elif team == 0:
			color = (0,22,200)
			cv2.putText(img, str(player), 
					(x, y), font, 0.5, color, 2)
		else:
			color = (255,102,0)
			cv2.circle(img, (x, y), 5, color, 2)
		print str(time)
		cv2.putText(img, "time: %s"%time, 
						(0,450), font, .5, (255,0,0), 2)
	return img

def path(file):
	p = np.genfromtxt(file, delimiter=',', dtype=int)
	xp, fp1, fp2 = p[:,0], p[:,1], p[:,2]
	t = range(min(xp), max(xp))
	x = np.interp(t, xp, fp1)
	y = np.interp(t, xp, fp2)
	return  np.vstack((t, x, y)).T

pos = np.random.randint(10, high=450, size=(50, 5, 2))
#pos = np.array([[60, 300], [80, 310], [55, 280], [85, 290], [61, 301]])
#pos = np.genfromtxt("pos.csv", delimiter=',', skip_header=1, dtype=int).reshape(-1, 5, 5)

#plt.imshow(background())
#plt.grid()
#plt.savefig('court.png')
#plt.show()
p1 = path("player1.csv")
p2 = path("player2.csv")
p3 = path("player3.csv")
p4 = path("player4.csv")
ball = path("ball.csv")
#import pdb; pdb.set_trace()

for (t, x1, y1, _, x2, y2, _, x3, y3, _, x4, y4, _, x5, y5) in np.hstack((p1,p2,p3,p4,ball)):
	bg = background()
	cv2.putText(bg, "1", (int(x1), int(y1)), font, 0.5, (200,22,0), 2)
	cv2.putText(bg, "2", (int(x2), int(y2)), font, 0.5, (200,22,0), 2)
	cv2.putText(bg, "1", (int(x3), int(y3)), font, 0.5, (0,22,200), 2)
	cv2.putText(bg, "2", (int(x4), int(y4)), font, 0.5, (0,22,200), 2)
	cv2.circle(bg, (int(x5), int(y5)), 5, (255,102,0), -1)
	cv2.putText(bg, "time: %s"%t, 
						(0,450), font, .5, (255,0,0), 2)
	cv2.imshow('img',bg)
	#import pdb; pdb.set_trace()
	if cv2.waitKey(50) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()