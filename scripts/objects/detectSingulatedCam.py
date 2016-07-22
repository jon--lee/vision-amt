from PIL import Image
import numpy as np
import math
import time
from Net.tensor import inputdata
from pipeline.bincam import BinaryCamera
import cv2

def numpify(arr):
    subs = []
    for sub in arr:
        npSub = np.array(sub)
        subs.append(npSub)
    return np.array(subs)
class Group:
    #must be more than 10 pixels apart to be singulated
    tol = 20
    tolSquared = 400
    def __init__(self):
        self.edges = []
        self.points = []
        self.lowX = -1
        self.highX = -1
        self.lowY = -1
        self.highY = -1
        self.area = 0
    def squaredDist(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    def isAdj(p1, p2):
        xAdj = p1[0] == p2[0] or p1[0] - 1 == p2[0] or p1[0] + 1 == p2[0]
        yAdj = p1[1] == p2[1] or p1[1] - 1 == p2[1] or p1[1] + 1 == p2[1]
        return xAdj and yAdj
    #should never be called before any points are added
    def containsPoint(self, p):
        #basic check for far away points
        if p[0] > self.highX + Group.tol or p[0] < self.lowX - Group.tol or p[1] > self.highY + Group.tol or p[1] < self.lowY - Group.tol:
            return False
        for edg in self.edges:
            e = edg[0]
            if Group.squaredDist(p, e) < Group.tolSquared:
                return True
        return False
    def merge(self, other):
        self.lowX = min(self.lowX, other.lowX)
        self.lowY = min(self.lowY, other.lowY)
        self.highX = max(self.highX, other.highX)
        self.highY = max(self.highY, other.highY)
        self.area = self.area + other.area
        self.points = self.points + other.points
        self.edges = self.edges + other.edges
    def add(self, p):
        self.points.append(p)
        count = 0
        currEdges = []
        for e in self.edges:
            if Group.isAdj(e[0], p):
                e[1] += 1
                count += 1
                #8 adjacent blocks = surrounded, not an edge
                #8 in theory, but actually works with 2- sparse edge points
                #work with 0 for certain shapes, but not for L-shaped components
                numAdjEdges = 2 #change to 8 if not working for certain shapes
                if e[1] < numAdjEdges:
                    currEdges.append(e)
            else:
                currEdges.append(e)
        self.edges = currEdges
        if count < 8:
            self.edges.append([p, count])
        if p[0] > self.highX or self.highX == -1:
            self.highX = p[0]
        if p[0] < self.lowX or self.lowX == -1:
            self.lowX = p[0]
        if p[1] > self.highY or self.highY == -1:
            self.highY = p[1]
        if p[1] < self.lowY or self.lowY == -1:
            self.lowY = p[1]
        self.area += 1
    def getPoints(self):
        return self.points
def highlight(mat):
    groups = []
    imgW = 420
    imgH = 420
    for x in range(imgW):
        for y in range(imgH):
            if mat[y][x][0] > 200 and mat[y][x][1] < 100 and mat[y][x][2] < 100:
                p = (x, y)
                adjGroups = []
                nonAdjGroups = []
                for g in groups:
                    if g.containsPoint(p):
                        adjGroups.append(g)
                    else:
                        nonAdjGroups.append(g)
                if len(adjGroups) == 0:
                    newG = Group()
                    newG.add(p)
                    nonAdjGroups.append(newG)
                else:
                    adjGroups[0].add(p)
                    for i in range(1, len(adjGroups)):
                        adjGroups[0].merge(adjGroups[i])
                    nonAdjGroups.append(adjGroups[0])
                groups = nonAdjGroups
    colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255)]
    colorsT = [(0, 255, 0, 255), (0, 0, 255, 255), (0, 255, 255, 255), (255, 0, 255, 255), (255, 255, 0)]
    it = 0
    for g in groups:
        colNum = it % len(colors)
        for p in g.getPoints():
            x, y = p[0], p[1]
            mat[y][x] = colorsT[colNum]
        it += 1
    return mat
bc = BinaryCamera("./meta.txt")
bc.open()
frame = bc.read_frame()
try:
    while True:
        frame = bc.read_frame()
        out = highlight(frame)
        cv2.imshow("camera",out)
        cv2.waitKey(20)
except KeyboardInterrupt:
    pass
