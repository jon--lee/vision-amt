from PIL import Image
import numpy as np
import math
import time
"""converts 2D python list to 2D numpy array"""
def numpify(pyList):
    npSubarrays = []
    for pySublist in pyList:
        npSubarray = np.array(pySublist)
        npSubarrays.append(npSubarray)
    return np.array(npSubarrays)
"""groups pixels into object clusters"""
class Group:
    # must be more than 'tol' pixels apart to be singulated
    tol = 20
    tolSquared = tol**2
    dim = 2
    def __init__(self):
        self.edges = []
        self.points = []
        self.lowCoords = [-1 for i in range(Group.dim)]
        self.highCoords = [-1 for i in range(Group.dim)]
        self.area = 0
    def squaredDist(p1, p2):
        return sum([(p1[i] - p2[i])**2 for i in range(Group.dim)])
    def isAdj(p1, p2):
        adjacent = True
        for i in range(Group.dim):
            adjacent = adjacent and (p1[i] == p2[i] or p1[i] - 1 == p2[i] or p1[i] + 1 == p2[i])
        return adjacent
    """determines if point is close enough to group to be added"""
    def pointInRange(self, p):
        #should never be called before any points are added
        if len(self.points) == 0:
            return False
        #basic check for points outside outmost range
        basicCheck = False
        for i in range(0, Group.dim):
            basicCheck = basicCheck or p[i] > self.highCoords[i] + Group.tol or p[i] < self.lowCoords[i] - Group.tol
        if basicCheck:
            return False
        #accurate check for distance to each edge
        for edge in self.edges:
            edgeCoord = edge[0]
            if Group.squaredDist(p, edgeCoord) < Group.tolSquared:
                return True
        return False
    def mergeGroups(self, other):
        for i in range(0, Group.dim):
            self.lowCoords[i] = min(self.lowCoords[i], other.lowCoords[i])
            self.highCoords[i] = max(self.highCoords[i], other.highCoords[i])
        self.area = self.area + other.area
        self.points = self.points + other.points
        self.edges = self.edges + other.edges
    def addPoint(self, p):
        self.points.append(p)
        numAdjPoints = 0
        currEdges = []
        for edge in self.edges:
            if Group.isAdj(edge[0], p):
                edge[1] += 1
                numAdjPoints += 1
                #8 adjacent blocks = surrounded, not an edge
                #actually works with less than 8 for most shapes- sparse edge points
                #increase if not working for certain shapes
                numAdjEdges = 2
                if edge[1] < numAdjEdges:
                    currEdges.append(e)
            else:
                #always keep nonadjacent edges
                currEdges.append(edge)
        self.edges = currEdges
        if numAdjPoints < 8:
            self.edges.append([p, numAdjPoints])
        #update shape boundaries
        for i in range(0, Group.dim):
            if self.lowCoords[i] > p[i] or self.lowCoords[i] == -1:
                self.lowCoords[i] = p[i]
            if self.highCoords[i] < p[i] or self.highCoords[i] == -1:
                self.highCoords[i] = p[i]
        self.area += 1
    def getPoints(self):
        return self.points
    def getEdges(self):
        return self.edges
img = Image.open("rect.png")
imgW, imgH = img.size[0], img.size[1]
mat = [[0 for i in range(imgW)] for j in range(imgH)]
for x in range(imgW):
    for y in range(imgH):
        mat[y][x] = img.getpixel((x, y))
groups = []
stTime = time.time()
for x in range(imgW):
    for y in range(imgH):
        if mat[y][x][0] > 200 and mat[y][x][1] < 100 and mat[y][x][2] < 100:
            p = (x, y)
            adjGroups = []
            nonAdjGroups = []
            for g in groups:
                if g.pointInRange(p):
                    adjGroups.append(g)
                else:
                    nonAdjGroups.append(g)
            if len(adjGroups) == 0:
                newG = Group()
                newG.addPoint(p)
                nonAdjGroups.append(newG)
            else:
                adjGroups[0].addPoint(p)
                for i in range(1, len(adjGroups)):
                    adjGroups[0].mergeGroups(adjGroups[i])
                nonAdjGroups.append(adjGroups[0])
            groups = nonAdjGroups
print(time.time() - stTime)
colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255)]
colorsT = [(0, 255, 0, 255), (0, 0, 255, 255), (0, 255, 255, 255), (255, 0, 255, 255), (255, 255, 0)]
it = 0
# white = (255, 255, 255, 255)
# edgeList = []
for g in groups:
    # for e in g.getEdges():
    #     edgeList.append(e[0])
    colNum = it % len(colors)
    for p in g.getPoints():
        x, y = p[0], p[1]
        mat[y][x] = colorsT[colNum]
    it += 1
# countT = 0
# for e in edgeList:
#     countT += 1
#     x, y = e[0], e[1]
#     mat[y][x] = white
# print(countT)
#Colors largest group blue, singulated blocks green
img = Image.fromarray(np.uint8(numpify(mat)))
img.save("binEnd1.png")
