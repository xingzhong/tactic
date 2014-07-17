import pandas as pd
import numpy as np
from court import HOOPS
#import matplotlib.pyplot as plt
#from pylab import rand
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture.gmm import log_multivariate_normal_density

def polar(xy, zero=HOOPS[0]):
	x, y = xy-zero
	r = np.sqrt(x**2+y**2)
	theta = np.arctan2(y, x)
	return np.array([r, theta])

def DScore(pair1, pair2):
	means = np.array([50, 0]).reshape(1,2)
	covars = np.array([5, .1]).reshape(1,2)
	s1 = log_multivariate_normal_density(pair1, means, covars)
	s2 = log_multivariate_normal_density(pair2, means, covars)
	return pd.DataFrame(s1+s2, columns=['defence'])

def Degree(pl):
	v02 = pl[0][['x_m', 'y_m']] - pl[2][['x_m', 'y_m']]
	v12 = pl[1][['x_m', 'y_m']] - pl[2][['x_m', 'y_m']]
	v20 = pl[2][['x_m', 'y_m']] - pl[0][['x_m', 'y_m']]
	v30 = pl[3][['x_m', 'y_m']] - pl[0][['x_m', 'y_m']]
	n02 = np.linalg.norm(v02, axis=1)
	n12 = np.linalg.norm(v12, axis=1)
	n20 = np.linalg.norm(v20, axis=1)
	n30 = np.linalg.norm(v30, axis=1)
	nv2 = v02.div(n02, axis=0)
	nv0 = v20.div(n20, axis=0)
	nv2 = nv2.join(v12.div(n12, axis=0), rsuffix='_v')
	nv0 = nv0.join(v30.div(n30, axis=0), rsuffix='_v')
	degree2 = np.arccos(nv2.x_m * nv2.x_m_v + nv2.y_m * nv2.y_m_v)*57.3
	degree0 = np.arccos(nv0.x_m * nv0.x_m_v + nv0.y_m * nv0.y_m_v)*57.3
	degree0 = pd.DataFrame(degree0, columns=['degree0'])
	degree2 = pd.DataFrame(degree2, columns=['degree2'])
	#import pdb; pdb.set_trace()
	return degree0.join(degree2)


def visFeatures(PCs, fts):
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	pl = pd.Panel(PCs, minor_axis=['x', 'y'])
	fig = plt.figure()
	ax = fig.add_subplot(221, polar=True)
	bx = fig.add_subplot(223)
	cx = fig.add_subplot(224)
	Color = cm.jet

	for player_id in pl.major_axis:
		player = pl.major_xs(player_id).T
		playerMean = pd.rolling_mean(player, 3)
		playerPolar = playerMean.apply(polar, axis=1)
		ax.plot(playerPolar.y,	
					playerPolar.x, 
					color=Color(player_id/4.0),
					label=player_id)
	ax.legend(loc=(1,0))
	fts.defence.plot(ax=bx)
	fts[['degree0', 'degree2']].plot(ax=cx)
	plt.show()


def visCoord(PCs):
	pl = pd.Panel(PCs, minor_axis=['x', 'y'])
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Color = cm.jet
	for player_id in pl.major_axis:
		player = pl.major_xs(player_id).T
		playerMean = pd.rolling_mean(player, 3)
		ax.plot(playerMean[0],playerMean.index, playerMean[1], color=Color(player_id/4.0))
	N = pl.shape[0]
	ax.set_xlabel('r')
	ax.set_ylabel('t')
	ax.set_zlabel('theta')
	plt.show()

def features(PCs):
	pl = pd.Panel(PCs, minor_axis=['x', 'y'])
	players = []
	for player_id in pl.major_axis:
		player = pl.major_xs(player_id).T
		playerMean = pd.rolling_mean(player, 3)
		playerPolar = playerMean.apply(polar, axis=1)
		player = player.join(playerMean, rsuffix="_m")
		player = player.join(playerPolar, rsuffix="_p")
		players.append(player)
	newP = pd.Panel(dict(zip(range(4), players)))
	pair1 = (newP[0]-newP[1])[['x_p', 'y_p']]
	pair2 = (newP[2]-newP[3])[['x_p', 'y_p']]
	defence = DScore(pair1, pair2)
	degree = Degree(newP)
	return defence.join(degree)

if __name__ == '__main__':
	name = "../data/newPr"
	Ms = np.load("%s.ms.npy"%name)
	PCs = np.load("%s.players.npy"%name)
	fts = features(PCs)
	visFeatures(PCs, fts)
	#visPolar(pl)
	#import pdb; pdb.set_trace()
