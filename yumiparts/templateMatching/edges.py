from PIL import Image
import math
import time
import numpy as np
from scipy import signal

def numpify(arr):
    subs = []
    for sub in arr:
        npSub = np.array(sub)
        subs.append(npSub)
    return np.array(subs)
"""defining several convolution matrices / kernels"""
#uses https://en.wikipedia.org/wiki/Kernel_(image_processing)
gaussian = numpify([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
#uses https://en.wikipedia.org/wiki/Sobel_operator
sobelx = numpify([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = numpify([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
"""
converts from PIL image to matrix in [r][c] / [y][x] format
applies filter function to each pixel
"""
def getMatFromImg(img, filt):
    imgW, imgH = img.size[0], img.size[1]
    mat = [[0 for i in range(imgW)] for j in range(imgH)]
    for x in range(imgW):
        for y in range(imgH):
            pixCol = img.getpixel((x, y))
            mat[y][x] = filt(pixCol)
    return mat
"""
converts from matrix in [r][c] / [y][x] format to PIL image
"""
def getImgFromMat(mat):
    matW, matH = len(mat[0]), len(mat)
    img = Image.new("RGB", (matW, matH))
    for x in range(matW):
        for y in range(matH):
            img.putpixel((x, y), tuple(mat[y][x]))
    return img
"""
mat- matrix in [r][c] / [y][x] format containing RGB color tuples
filt- filter function to apply to each pixel
output: a new matrix with filtered pixels (original matrix is unchanged)"""
def applyToMat(mat, filt):
    matW, matH = len(mat[0]), len(mat)
    matFilt = [[0 for i in range(matW)] for j in range(matH)]
    for x in range(matW):
        for y in range(matH):
            filtCol = filt(mat, x, y)
            matFilt[y][x] = (int(filtCol[0]), int(filtCol[1]), int(filtCol[2]))
    return matFilt
"""
img - matrix in [r][c] / [y][x] format containing RGB color tuples
ker - 3x3 convolution matrix (kernel)
x, y - coordinates to get convolution of
output: 3 element RGB color tuple
"""
def getConvolution(img, ker, x, y):
    if (x - 1 < 0) or (x + 1 > len(img[0]) - 1) or (y - 1 < 0) or (y + 1 > len(img) - 1):
        return img[y][x]
    else:
        accum = [0, 0, 0]
        for yShift in range(0, 3):
            for xShift in range(0, 3):
                weight = ker[yShift][xShift]
                currX, currY = x - 1 + xShift, y - 1 + yShift
                prevPixVal = img[currY][currX]
                accum = [accum[i] + prevPixVal[i] * weight for i in range(3)]
        return (int(accum[0]), int(accum[1]), int(accum[2]))
"""
mat- matrix in [r][c] / [y][x] format containing RGB color tuples
x, y- coordinates of pixel to filter
applies a series of filters that make edges white and the rest of the image black
"""
def whiteEdges(mat, x, y):
    #apply sobel operator
    sobColX = getConvolution(mat, sobelx, x, y)
    sobColY = getConvolution(mat, sobely, x, y)
    sobCol = [math.sqrt(sobColX[i]**2 + sobColY[i]**2) for i in range(3)]
    #apply binary mask
    binCol = [255 if sobCol[i] > 128 else 0 for i in range(3)]
    #filter out non-white
    if all(rgbCol == 255 for rgbCol in binCol):
        binCol = [255, 255, 255]
    else:
        binCol = [0, 0, 0]
    return binCol
def gauss(mat, x, y):
    return getConvolution(mat, gaussian, x, y)
def decreaseSize(mat):
    s = time.time()
    mat = applyToMat(mat, gauss)
    print(time.time() - s)
    matW, matH = len(mat[0]), len(mat)
    outputMat = [[0 for i in range(math.floor(matW/2))] for j in range(math.floor(matH/2))]
    for x in range(math.floor(matW/2)):
        for y in range(math.floor(matH/2)):
            outputMat[y][x] = mat[y*2][x*2]
    return outputMat
def applyN(func, num, inp):
    if num == 1:
        return func(inp)
    else:
        return func(applyN(func, num - 1, inp))
def pyramid(mat):
    return applyN(decreaseSize, 2, mat)
def findMatch(temp, seg, startX, endX, startY, endY):
    minDiff = 9999999999999
    bestPos = [0, 0]
    for x in range(startX, endX):
        for y in range(startY, endY):
            diff = 0
            for xT in range(tempW):
                for yT in range(tempH):
                    pixSearch = pyramidSeg[y + yT][x + xT]
                    pixMatch = pyramidTemp[yT][xT]
                    diff += sum([abs(pixSearch[i] - pixMatch[i]) for i in range(3)])/3
            if diff < minDiff:
                minDiff = diff
                bestPos = [x, y]
    return bestPos
def traceObj(mat, pos, obj):
    def inOutline(mat, x, y):
        boundaryColor = (255, 0, 0)
        yRange = (pos[1] < y and y < (pos[1] + len(obj)))
        atX = (x == pos[0] or x == (pos[0] + len(obj[0])))
        if yRange and atX:
            return boundaryColor
        xRange = (pos[0] < x and x < pos[0] + len(obj[0]))
        atY = (y == pos[1] or y == (pos[1] + len(obj)))
        if xRange and atY:
            return boundaryColor
        return mat[y][x]
    return applyToMat(mat, inOutline)
segImgName = "edgeTest"
segImg = Image.open(segImgName + ".png")
segMat = getMatFromImg(segImg, lambda x: x)
s = time.time()
segMatFilt = applyToMat(segMat, whiteEdges)
print(time.time() - s)
s = time.time()
pyramidSeg = pyramid(segMatFilt)
print(time.time() - s)
#store list of matrices for template matching
templates = []
#create template outlines in same format as image to segment
for templateNum in range(1, 5):
    tempImg = Image.open("t" + str(templateNum) + ".png")
    transparentToBlack = lambda pixCol: (0, 0, 0) if pixCol[3] == 0 else (pixCol[0], pixCol[1], pixCol[2])
    tempMat = getMatFromImg(tempImg, transparentToBlack)
    tempMatFilt = applyToMat(tempMat, whiteEdges)
    templates.append(tempMatFilt)
    # tempImgRGB = getImgFromMat(tempMatFilt)
    # tempImgRGB.save("template" + str(templateNum) + ".png")
#based on https://en.wikipedia.org/wiki/Template_matching
testTemplate = templates[0]
pyramidTemp = pyramid(testTemplate)
segW, segH = len(pyramidSeg[0]), len(pyramidSeg)
tempW, tempH = len(pyramidTemp[0]), len(pyramidTemp)
s = time.time()
bestPos = findMatch(pyramidTemp, pyramidSeg, 0, segW - tempW, 0, segH - tempH)
print(time.time() - s)
#because of rounding, could check within a few pixels of bestPos for better match in original image
toDraw = traceObj(segMat, [bestPos[0]*4, bestPos[1]*4], testTemplate)
print("Object detected- can move")
#bestPos = findMatch(testTemplate, segmatFilt, bestPos[0] - 5, bestPos[0] + 5, bestPos[1] - 5, bestPos[1] + 5)
#toDraw = traceObj(segMat, bestPos, testTemplate)
#locate lines inside circle
#10 < x < 350
#180 - sqrt(170^2 - (x-180)^2) < y < 180 + sqrt(170^2 - (x-180)^2)
# for x in range(10, 351):
#     yTerm = int(math.sqrt(170**2 - (x-180)**2))
#     for y in range(180 - yTerm, 180 + yTerm + 1):
#         pass
        #c = out[y][x]
        #if all(c == 255 for r in res):

"""
todo: keep track of an object while it is moving
1: get its initial rectangular boundary (say when done)
2: as it moves, continously check for new matches within 5 or so pixels (not on shrunk version)
3: update bounding box
with multiple templates, compute distances between centers at every step and color black if far enough away
    #create some rectangular bounding boxes- if far enough from majority, color iinside black/purple

"""

#convert segmentation image to jpg
out = getImgFromMat(toDraw)
out.save(segImgName + "Out.png")
