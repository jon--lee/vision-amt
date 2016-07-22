import math
from PIL import Image

"""Stores original location of point and current value in feature space"""
class Point:
    def __init__(self, r, g, b):
        self.feature = [r, g, b]
    def getDim(self):
        return len(self.feature)
    def getFeature(self, i):
        return self.feature[i]
    def update(self, r, g, b):
        tempCopy = self.copy()
        self.feature = [r, g, b]
        return math.sqrt(self.squaredDist(tempCopy))
    def squaredDist(self, other):
        return sum([(self.feature[i] - other.feature[i])**2 for i in range(len(self.feature))])
    def getCol(self):
        return (self.feature[0], self.feature[1], self.feature[2])
    def gaussKernel(self, other):
        base = math.e
        c = 0.1
        exp = -1 * self.squaredDist(other) * c
        return math.pow(base, exp)
    def scale(self, weight):
        return [self.feature[i] * weight for i in range(len(self.feature))]
    def copy(self):
        RGB = self.getCol()
        return Point(RGB[0], RGB[1], RGB[2])
"""recomputes p based on the average of other ps, weighted by their feature distance"""
def weightedMean(p, ps, r, x, y, cps):
    p = cps[y][x]
    denomSum, numSum = 0, [0 for i in range(p.getDim())]
    for yi in range(y - r, y + r + 1):
        for xi in range(x - r, x + r + 1):
            if not ((x == xi and y == yi) or xi < 0 or xi >= len(ps[0]) or yi < 0 or yi >= len(ps)):
                pi = ps[yi][xi]
                weight = p.gaussKernel(pi)
                denomSum += weight
                numSum = [numSum[i] + weight * pi.getFeature(i) for i in range(p.getDim())]
    vals = [numSum[i]/denomSum for i in range(len(numSum))]
    return p.update(vals[0], vals[1], vals[2])
name, ext = "test", "jpg"
img = Image.open(name + "." + ext)
w, h = img.size[0], img.size[1]
originalPoints = [[0 for i in range(w)] for j in range(h)]
for x in range(w):
    for y in range(h):
        pix = img.getpixel((x, y))
        p = Point(pix[0], pix[1], pix[2])
        originalPoints[y][x] = p
#uses https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
#uses https://en.wikipedia.org/wiki/Mean_shift
searchRadius = 7
avgShift = 100
numDone = 0
while avgShift > 1:
    print(avgShift)
    copyPoints = [[p.copy() for p in row] for row in originalPoints]
    num, amt = 0, 0
    for y in range(len(copyPoints)):
        for x in range(len(copyPoints[0])):
            shift = weightedMean(p, originalPoints, searchRadius, x, y, copyPoints)
            amt += shift
            num += 1
    avgShift = amt/num
    originalPoints = copyPoints
print(avgShift)
for y in range(len(originalPoints)):
    for x in range(len(originalPoints[0])):
        p = originalPoints[y][x]
        rgb = p.getCol()
        img.putpixel((x, y), (int(rgb[0]), int(rgb[1]), int(rgb[2])))
img.save(name + "out." + ext)
