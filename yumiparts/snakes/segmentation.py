import numpy as np
from PIL import Image, ImageFilter
import math
import time

def toGray(RGB):
    return 0.3 * RGB[0] + 0.6 * RGB[1] + 0.1 * RGB[2]
def compose(f, g):
    return lambda x: f(g(x))
def apply3(f, a1, a2, a3):
    return f(a1), f(a2), f(a3)
def getValidSquares(cp, height, width):
    x, y = cp.p[0], cp.p[1]
    validSquares = []
    for cx in range(x - 4, x + 5):
        if cx >= 0 and cx < width:
            for cy in range(y - 3, y + 4):
                if cy >= 0 and cy < height:
                    validSquares.append([cx, cy])
    return validSquares
def copy(s):
    return [s[0], s[1]]
def avgD(snakePoints):
    d = 0
    numP = len(snakePoints)
    for cp in snakePoints:
        d += cp.squaredDist(cp.nex.p[0], cp.nex.p[1])
    avgD = d/numP
    return avgD
#helper function for gradient
def getAdjacent(coord, minVal, maxVal, prev):
    if prev:
        val = coord - 1
        if val < minVal:
            val = maxVal
        return val
    else:
        val = coord + 1
        if val > maxVal:
            val = minVal
        return val
#returns squared difference between pixel before and pixel after (wraps around edges)
def getGrad(m, p, axis):
    x, y = p[0], p[1]
    maxX, maxY = len(m[0]) - 1, len(m) - 1
    val1 = m[y][getAdjacent(x, 0, maxX, True)]
    val2 = m[y][getAdjacent(x, 0, maxX, False)]
    if (axis == 1):
        val1 = m[getAdjacent(y, 0, maxY, True)][x]
        val2 = m[getAdjacent(y, 0, maxY, False)][x]
    return (val2 - val1) ** 2
def dot(v0, v1):
    return v0[0] * v1[0] + v0[1] * v1[1]
def normalize(v):
    norm = math.sqrt(v[0]**2 + v[1]**2)
    if norm == 0:
        return v
    return [v[0]/norm, v[1]/norm]
class ContourP:
    def __init__(self, x, y):
        self.p = [x, y]
        self.prev, self.nex = None, None
    def setNext(self, other):
        if self.nex == None:
            self.nex = other
            other.prev = self
    def update(self, x, y):
        self.p = [x, y]
    """returns all pixels between two adjacent points"""
    def getPoints(self):
        if self.nex != None:
            # create mutable copies
            sG, fG = copy(self.p), copy(self.nex.p)
            points = []
            dG = [fG[i] - sG[i] for i in range(len(sG))]
            def update(xyTo):
                nonlocal sG, fG, dG, xyBack
                sG, fG, dG = apply3(xyTo, sG, fG, dG)
                xyBack = compose(xyBack, xyTo)
            # special cases for vertical and horizontal lines
            if dG[0] == 0 or dG[1] == 0:
                #transform to case of dG[0] == 0 and dG[1] > 0
                xyBack = lambda p: p
                if dG[1] == 0:
                    update(lambda p: [p[1], p[0]])
                if dG[1] < 0:
                    update(lambda p: [p[0], -1 * p[1]])
                while sG[1] < fG[1]:
                    points.append(xyBack(copy(sG)))
                    sG[1] += 1
            # special case for lines with slope magnitude = 1
            elif abs(dG[1]/dG[0]) == 1:
                #transform to case of dG[1]/dG[0] == 1 and dG[1] > 0
                xyBack = lambda p: p
                if dG[1]/dG[0] == -1:
                    update(lambda p: [p[0], -1 * p[1]])
                if dG[1] < 0:
                    update(lambda p: [p[0] * -1, p[1] * -1])
                while sG[1] < fG[1]:
                    points.append(xyBack(copy(sG)))
                    sG[0] += 1
                    sG[1] += 1
            else:
                #algorithm from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
                #transform to 0th octant: 0 < dG[1]/dG[0] < 1 and dG[1] > 0
                xyBack = lambda p: p
                if abs(dG[1]/dG[0]) > 1:
                    update(lambda p: [p[1], p[0]])
                #0, 3, 4, or 7 remain
                if dG[1]/dG[0] < 0:
                    update(lambda p: [p[0], -1 * p[1]])
                #0 or 4 remain
                if dG[1] < 0:
                    update(lambda p: [p[0] * -1, p[1] * -1])
                #Bresenham's line algorithm for 1st octant:
                D = dG[1] - dG[0]
                while sG[0] < fG[0]:
                    points.append(xyBack(copy(sG)))
                    if D >= 0:
                        sG[1] += 1
                        D -= dG[0]
                    D += dG[1]
                    sG[0] += 1
            return points
    def squaredDist(self, x, y):
        return (self.p[0] - x)**2 + (self.p[1] - y)**2
    """gradients are in the form [y][x], square is in the form of (x, y)"""
    def calcEnergy(self, s, xGrad, yGrad, avgD):
        # uses http://www.cse.unr.edu/~bebis/CS791E/Notes/DeformableContours.pdf
        newX, newY = s[0], s[1]
        contE = (avgD - self.prev.squaredDist(newX, newY))**2
        smoothE = (self.prev.p[0] - 2 * newX + self.nex.p[0])**2 + (self.prev.p[1] - 2 * newY + self.nex.p[1])**2
        v1 = [self.prev.p[0] - newX, self.prev.p[1] - newY]
        v1 = normalize(v1)
        v2 = [self.nex.p[0] - newX, self.nex.p[1] - newY]
        v2 = normalize(v2)
        dp = v1[0] * v2[0] + v1[1] * v2[1]
        if dp < -1:
            dp = -1
        if dp > 1:
            dp = 1
        theta = math.acos(dp)
        gradE = xGrad[newY][newX] * abs(math.cos(theta)) + yGrad[newY][newX] * abs(math.sin(theta))
        return contE + smoothE - 2 * gradE
stTime = time.time()
name = "test"
#convert color jpg to grayscale matrix in [r][c] or [y][x] format
img = Image.open(name + ".jpg")
blurred = img.filter(ImageFilter.BLUR)
w, h = blurred.size[0], blurred.size[1]
im = [[0 for i in range(w)] for j in range(h)]
realIm = [[0 for i in range(w)] for j in range(h)]
for x in range(w):
    for y in range(h):
        im[y][x] = toGray(blurred.getpixel((x, y)))
        realIm[y][x] = img.getpixel((x, y))
#precompute x and y gradient maps
xGrad, yGrad = [], []
for i in range(len(im)):
    r = im[i]
    newRX, newRY = [], []
    for j in range(len(r)):
        gx, gy = getGrad(im, (j, i), 0), getGrad(im, (j, i), 1)
        newRX.append(gx)
        newRY.append(gy)
    xGrad.append(newRX)
    yGrad.append(newRY)
for xi in range(0, int(w/50)):
    lowX = xi * 50
    highX = lowX + 50
    for yi in range(0, int(h/50)):
        lowY = yi * 50
        highY = lowY + 50
        #rectangular bounding box with jrange * irange points in range [low, high] x [low, high]
        snakePoints = []
        firstP = ContourP(lowX, lowY)
        snakePoints.append(firstP)
        oldP = firstP
        pointsPerSide = 10
        stepX, stepY = int((highX - lowX)/pointsPerSide), int((highY - lowY)/pointsPerSide)
        for j in range(0, 4):
            for i in range(0, pointsPerSide):
                if not (i == 0 and j == 0):
                    if j == 0:
                        newP = ContourP(i * stepX + lowX, lowY)
                    elif j == 1:
                        newP = ContourP(highX, i * stepY + lowY)
                    elif j == 2:
                        newP = ContourP(highX - i * stepX, highY)
                    else:
                        newP = ContourP(lowX, highY - i * stepY)
                    snakePoints.append(newP)
                    oldP.setNext(newP)
                    oldP = newP
        oldP.setNext(firstP)
        # snake contour using algorithm from http://www.cs.utah.edu/~manasi/coursework/cs7960/p5/project5.html
        numIters, maxIters = 0, 20
        while numIters < maxIters:
            for contourPoint in snakePoints:
                squares = getValidSquares(contourPoint, h, w)
                minE = float('inf')
                minS = None
                for s in squares:
                    e = contourPoint.calcEnergy(s, xGrad, yGrad, avgD(snakePoints))
                    if e < minE:
                        minE = e
                        minS = s
                contourPoint.update(minS[0], minS[1])
            numIters += 1
        #mark segmented boundary
        for contourPoint in snakePoints:
            allPoints = contourPoint.getPoints()
            for point in allPoints:
                im[point[1]][point[0]] = 255
                realIm[point[1]][point[0]] = 255
#convert grayscale matrix to grayscale jpg
gIm = Image.new("P", (w, h))
rgbIm = Image.new("RGB", (w, h))
for x in range(w):
    for y in range(h):
        gIm.putpixel((x, y), im[y][x])
        rgbIm.putpixel((x, y), realIm[y][x])
gIm.save(name + "g.png")
rgbIm.save(name + "rgb.png")
print(time.time() - stTime)
