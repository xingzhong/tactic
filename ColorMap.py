import cv2
import numpy as np

class ColorMap:
    startcolor = ()
    endcolor = ()
    startmap = 0
    endmap = 0
    colordistance = 0
    valuerange = 0
    ratios = []    

    def __init__(self, startcolor, endcolor, startmap, endmap):
        self.startcolor = np.array(startcolor)
        self.endcolor = np.array(endcolor)
        self.startmap = float(startmap)
        self.endmap = float(endmap)
        self.valuerange = float(endmap - startmap)
        self.ratios = (self.endcolor - self.startcolor) / self.valuerange

    def __getitem__(self, value):
        color = tuple(self.startcolor + (self.ratios * (value - self.startmap)))
        return (int(color[0]), int(color[1]), int(color[2]))

if __name__ == '__main__':
    cmap = ColorMap((0,200,0), (180,0,200), 0, 10)
    blank = np.zeros((256,256,3))
    for i in range(10):
        color = cmap[i]
        print color
        cv2.circle(blank, (20*i,20*i), 5, color, -1)
    while(True):
        cv2.imshow('frame', blank)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    