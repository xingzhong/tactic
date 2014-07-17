## do court mapping 

import numpy as np
import cv2
from court import template, POI, HOOPS
from collections import deque
import itertools
from guppy import hpy
from features import features

COLORS = [(255,0,0), (0,255,0), (0,255,255), (255,255,0)]
font = cv2.FONT_HERSHEY_SIMPLEX
drawing = -1


def clickCourt(event, x, y, flags, param):
	img, num, courtPts, rawPts = param
	if event == cv2.EVENT_LBUTTONDBLCLK:
		distances = np.linalg.norm(np.array([x,y]) - POI, axis=1)
		if min(distances) < 15 :
			x, y = POI[np.argmin(distances)]
		courtPts.append((x, y))
		rawPts.append((x, y))

def clickRaw(event, x, y, flags, param):
	global drawing
	img, num, courtPts, rawPts, players = param
	if event == cv2.EVENT_LBUTTONDOWN:
		if len(players) > 0 :
			tgt = np.vstack((np.array(rawPts), np.array(players)))
		else:
			tgt = np.array(rawPts)
		distances = np.linalg.norm(np.array([x,y]) - tgt, axis=1)
		if min(distances) < 15 :
			drawing = np.argmin(distances)
			if drawing < len(rawPts):
				rawPts[drawing] = (x, y)
			else:
				players[drawing-len(rawPts)] = (x, y)
			
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing > -1:
			if drawing < len(rawPts):
				rawPts[drawing] = (x, y)
			else:
				players[drawing-len(rawPts)] = (x, y)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = -1

	elif event == cv2.EVENT_LBUTTONDBLCLK:
		players.append((x, y))

def pts_span(pt):
	x, y = pt
	return [ (x-1, y), (x, y-1), (x, y), (x, y+1), (x+1, y)]

def auto_mapping(frame, tpl, courtPts, rawPts, players):
	#import pdb; pdb.set_trace()
	if len(courtPts) == 4:
		frame_h, frame_w, _ = frame.shape
		edgeRaw = cv2.Canny(frame,100,200)
		edgeRaw = cv2.bitwise_and(edgeRaw, edgeRaw)
		edgeCrot = np.zeros_like(edgeRaw)
		total = float(np.sum(edgeRaw))
		cpts = np.float32(courtPts)
		cRot = np.zeros_like(frame)
		argmax, maxNum = None, 0
		#import pdb; pdb.set_trace()
		for idx, pts in enumerate(itertools.product(*map(pts_span, rawPts))):
			m = cv2.getPerspectiveTransform(cpts, np.float32(pts))
			cv2.warpPerspective(tpl, m, (frame_w, frame_h), dst=cRot)
			cv2.Canny(cRot,100,200, edges=edgeCrot)
			cv2.bitwise_and(edgeRaw, edgeCrot, dst=edgeCrot)
			score = np.sum(edgeCrot)
			if score > maxNum:
				argmax = pts
				maxNum = score
		
		print pts
		for i in range(4):
			rawPts[i] = pts[i]
		
		


def sub_mapping(frame, tpl, num, courtPts=[], rawPts=[], players=[]):
	frame_h, frame_w, _ = frame.shape
	edgeRaw = cv2.Canny(frame,100,200)
	oldM = np.zeros((3,3), dtype=np.float32)
	cRot = np.zeros_like(frame)
	cPlayers = []
	score = 0
	cv2.setMouseCallback('court', clickCourt, param=(tpl, num, courtPts, rawPts))
	cv2.setMouseCallback('dst', clickRaw, param=(frame, num, courtPts, rawPts, players))
	court, raw = tpl.copy(), frame.copy()
	while(1):
		cv2.copyMakeBorder(tpl, 0,0,0,0,0, dst=court)
		cv2.copyMakeBorder(frame, 0,0,0,0,0, dst=raw)
		for (x, y) in POI:
			cv2.circle(court, (x, y), 2, (0,0,255), -1)
		for (x, y) in HOOPS:
			cv2.circle(court, (x, y), 2, (0,0,255), -1)
		for idx, (x,y) in enumerate(courtPts):
			cv2.circle(court, (x, y), 5, COLORS[idx], 2)

		for idx, player in enumerate(players):
			cv2.ellipse(raw, player, (25, 15), 0, 0, 360, COLORS[idx], 2)
			cv2.circle(raw, player, 3, COLORS[idx], -1)
			cv2.line(raw, player, (player[0]+25, player[1]), COLORS[idx], 2)
		
		if len(courtPts) == 4 :
			M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(rawPts))
			N = cv2.getPerspectiveTransform(np.float32(rawPts), np.float32(courtPts))
			if np.linalg.norm(M-oldM)>0.01:
				print "call warpPerspective"
				cv2.warpPerspective(tpl, M, (frame_w, frame_h), dst=cRot)
				edgeCrot = cv2.Canny(cRot,100,200)
				imgAnd = cv2.bitwise_and(edgeRaw, edgeCrot)
				score = np.sum(imgAnd)/float(np.sum(edgeRaw))
				oldM = M
			cv2.addWeighted(cRot, 0.3, raw, 0.7, 0, dst=raw)
			if len(players)>0:
				cPlayers = cv2.perspectiveTransform(np.float32(players).reshape(-1,1,2), N)
				cPlayers = cPlayers.reshape(-1,2).astype(int)
		
		for idx, (x,y) in enumerate(rawPts):
			cv2.circle(raw, (x, y), 6, COLORS[idx], 3)

		for idx, player in enumerate(cPlayers):
			cv2.circle(court, tuple(player), 5, COLORS[idx], -1)

		cv2.putText(raw, "# %s"%num, (30,30), font, 1, (255,255,255), 1)
		cv2.putText(raw, "S %.2f %%"%(100*score), (30,70), font, 1, (255,255,255), 1)
		cv2.imshow('dst', raw)
		cv2.imshow('court', court)
		k = cv2.waitKey(20) & 0xFF 
		if k == ord('c'):
			return M, cPlayers
		elif k == ord('q'):
			return 
		elif k == ord('a'):
			print rawPts
			auto_mapping(frame, tpl, courtPts, rawPts, players)
			print rawPts

def mapping(name='../nba4_clip'):
	cv2.namedWindow('dst')
	cv2.namedWindow('court')
	cap = cv2.VideoCapture("%s.avi"%name)
	#cap = cv2.VideoCapture('../nba4_clip.avi')
	num = 1
	Ms, courtPts, rawPts, players, playersInCourt = [], [], [], [], []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		tpl = template()
		courtPts, rawPts = deque(courtPts, maxlen=4), deque(rawPts, maxlen=4)
		players = deque(players, maxlen=4)
		M, cPlayers = sub_mapping(frame, tpl, num, 
					courtPts=courtPts, rawPts=rawPts, players=players)
		if M is None:
			break
		Ms.append(M)
		playersInCourt.append(cPlayers)
		for pt, cp in enumerate(playersInCourt[::-1]):
			strong = int(2+3*np.exp(-pt))
			for idx, player in enumerate(cp):
				cv2.circle(tpl, tuple(player), strong, COLORS[idx], -1)
		cv2.imshow('route', tpl)
		num += 1
	Ms = np.array(Ms)
	playersInCourt = np.array(playersInCourt)
	np.save('%s.ms'%name, Ms)
	np.save('%s.players'%name, playersInCourt)
	cap.release()
	cv2.destroyAllWindows()
		

def show(name="../nba4_clip"):
	cap = cv2.VideoCapture("%s.avi"%name)
	Ms = np.load('%s.ms.npy'%name)
	PCs = np.load('%s.players.npy'%name)
	fts = features(PCs)
	MM, _, _ = Ms.shape
	num = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		tpl = template()
		frame_h, frame_w, _ = frame.shape
		cRot = cv2.warpPerspective(tpl, Ms[num], (frame_w, frame_h))
		frame = cv2.addWeighted(cRot, 0.4, frame, 0.6, 0)
		cv2.putText(tpl, "# %s"%num, (50,40), font, .5, (0,0,255), 2)
		cv2.putText(tpl, "Def %s"%fts.ix[num].defence, (50,60), font, .5, (0,0,255),2)
		cv2.putText(tpl, "Deg %s"%fts.ix[num].degree0, (50,80), font, .5, (0,0,255), 2)
		cv2.putText(tpl, "Deg %s"%fts.ix[num].degree2, (50,100), font, .5, (0,0,255), 2)
		#cv2.imshow('frame', frame)
		for idx, player in enumerate(PCs[num]):
			cv2.circle(tpl, tuple(player), 6, COLORS[idx], 4)
			cv2.line(tpl, tuple(player), tuple(HOOPS[0]),(255,255,255), 1 )
		cv2.line(tpl, tuple(PCs[num][0]), tuple(PCs[num][2]), (255,255,255), 2)
		cv2.line(tpl, tuple(PCs[num][1]), tuple(PCs[num][2]), (255,255,255), 2)
		cv2.line(tpl, tuple(PCs[num][3]), tuple(PCs[num][0]), (255,255,255), 2)
		cv2.imshow('court', tpl)
		k = cv2.waitKey(200) & 0xFF 
		if k == ord('q') or num > MM-2:
			break
		num += 1
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	#mapping(name="../data/newPr")
	#player()
	show(name="../data/newPr")