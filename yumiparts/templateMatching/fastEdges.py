from PIL import Image
import math
import time
import numpy as np
from scipy import signal
from skimage.measure import block_reduce

"""
Takes a 2d python list and converts it to a 2d numpy array
"""
def numpify(arr):
    subs = []
    for sub in arr:
        npSub = np.array(sub)
        subs.append(npSub)
    return np.array(subs)
"""
converts from PIL image to 2d numpy array in [r][c] format
applies filter function to each pixel
"""
def getMatFromImg(img, filt):
    imgW, imgH = img.size[0], img.size[1]
    mat = [[0 for i in range(imgW)] for j in range(imgH)]
    for x in range(imgW):
        for y in range(imgH):
            pixCol = img.getpixel((x, y))
            mat[y][x] = filt(pixCol)
    mat = numpify(mat)
    return mat
"""
converts from 2d numpy array in [r][c] format with grayscale values to PIL image
"""
def getImgFromMat(mat):
    m2 = np.copy(mat)
    m3 = np.copy(mat)
    rgb = np.stack((mat, m2, m3), axis = -1)
    img = Image.fromarray(np.uint8(rgb))
    return img
"""defining several convolution kernels"""
#uses https://en.wikipedia.org/wiki/Kernel_(image_processing)
gaussian = numpify([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
#uses https://en.wikipedia.org/wiki/Sobel_operator
sobelx = numpify([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.transpose(sobelx)
"""
mat- 2d numpy array in [r][c]
applies a series of filters that make edges white and the rest of the image black
"""
def whiteEdges(mat):
    #apply sobel operator
    sobColX = signal.fftconvolve(mat, sobelx, mode = "same")
    sobColY = signal.fftconvolve(mat, sobely, mode = "same")
    sobCol = np.hypot(sobColX, sobColY)
    return sobCol
def decreaseSize(mat):
    mat = signal.fftconvolve(mat, gaussian, mode = "same")
    outputMat = block_reduce(mat, block_size = (2, 2), func = np.mean)
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
    tempW, tempH = len(temp[0]), len(temp)
    for x in range(startX, endX):
        for y in range(startY, endY):
            diff = 0
            for xT in range(tempW):
                for yT in range(tempH):
                    pixSearch = seg[y + yT][x + xT]
                    pixMatch = temp[yT][xT]
                    diff += abs(pixSearch - pixMatch)
            if diff < minDiff:
                minDiff = diff
                bestPos = [x, y]
    return bestPos
def traceObj(mat, pos, obj):
    boundaryColor = 255
    copyMat = np.copy(mat)
    for y in range(pos[1], pos[1] + len(obj)):
        copyMat[y][pos[0]] = boundaryColor
        copyMat[y][pos[0] + len(obj[0])] = boundaryColor
    for x in range(pos[0], pos[0] + len(obj[0])):
        copyMat[pos[1]][x] = boundaryColor
        copyMat[pos[1] + len(obj)][x] = boundaryColor
    return copyMat
segImgName = "edgeTest2"
segImg = Image.open(segImgName + ".png")
#remove alpha channel and make rgb into grayscale
toGray = lambda x: 0.3 * x[0] + 0.6 * x[1] + 0.1 * x[2]
segMat = getMatFromImg(segImg, toGray)
segMatFilt = whiteEdges(segMat)
pyramidSeg = pyramid(segMatFilt)
#store list of matrices for template matching
templates = []
#create template outlines in same format as image to segment
for templateNum in range(1, 5):
    tempImg = Image.open("t" + str(templateNum) + ".png")
    toGrayWithTransparent = lambda x: 0 if x[3] == 0 else toGray(x)
    tempMat = getMatFromImg(tempImg, toGrayWithTransparent)
    tempMatFilt = whiteEdges(tempMat)
    templates.append(tempMatFilt)
#based on https://en.wikipedia.org/wiki/Template_matching
i = 0
for testTemplate in templates:
    pyramidTemp = pyramid(testTemplate)
    segW, segH = len(pyramidSeg[0]), len(pyramidSeg)
    tempW, tempH = len(pyramidTemp[0]), len(pyramidTemp)
    bestPos = findMatch(pyramidTemp, pyramidSeg, 0, segW - tempW, 0, segH - tempH)
    bestPos = [bestPos[0] * 4, bestPos[1] * 4]
    if i == 0:
        toDraw = traceObj(segMat, bestPos, testTemplate)
    else:
        toDraw = traceObj(toDraw, bestPos, testTemplate)
    print("iter")
    i += 1
    # print("Object detected- can move")
    # dist = 3
    # segMat = getMatFromImg(segImg, toGray)
    # segMatFilt = whiteEdges(segMat)
    # bestPos = findMatch(testTemplate, segMatFilt, bestPos[0] - dist, bestPos[0] + dist, bestPos[1] - dist, bestPos[1] + dist)

#convert segmentation image to jpg
out = getImgFromMat(toDraw)
out.save(segImgName + "Out.png")
