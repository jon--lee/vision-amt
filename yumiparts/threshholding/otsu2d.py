""""
TODO:
allow 1D and 2D versions (pixel, pixel and mean)
allow multiple threshholds (black, gray, white)
how to separate around point (not just line with m = -1)
document otsu method better
"""


from PIL import Image
import numpy as np
import time
"""
converts from python list to numpy array in 2D
"""
def numpify2DList(pyList):
    npSubs = []
    for pySub in pyList:
        npSub = np.array(pySub)
        npSubs.append(npSub)
    return np.array(npSubs)
"""
converts RGB value to single grayscale value
if value is already grayscale, return it
see: http://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
"""
def toGray(pix):
    weights = np.array([0.3, 0.6, 0.1])
    return np.dot(np.array(pix[0:3]), weights) if type(pix) != int else pix
"""
Otsu's Algorithm
finds threshhold value to split histogram into two classes
maximize inter-class variance / minimize intra-class variance
see: https://en.wikipedia.org/wiki/Otsu%27s_method
"""
def otsu(hist, L, totalNum):
    valSum = 0
    for pixVal in range(L):
        valSum += pixVal * hist[pixVal]
    valBelow, numBelow, numAbove, meanBelow, meanAbove = 0, 0, 0, 0, 0
    theMax, between, thresh1, thresh2 = 0, 0, 0, 0
    for currVal in range(L):
        numBelow += hist[currVal]
        if numBelow == 0:
            continue
        numAbove = totalNum - numBelow
        if numAbove == 0:
            break
        valBelow += currVal * hist[currVal]
        meanBelow = valBelow / numBelow
        meanAbove = (valSum - valBelow) / numAbove
        between = numBelow * numAbove * (meanBelow - meanAbove)**2
        if between >= theMax:
            thresh1 = currVal
            if between > theMax:
                thresh2 = currVal
            theMax = between
    thresh = (thresh1 + thresh2)/2
    return thresh
"""
converts rgb image to black and white image
uses 3D threshholding method for classification robust to noise
dimensions: pixel value, mean neighboring pixel value, median neighboring pixel value
O(L), not O(L^3), because it is performs 3 1D classifications, not 1 3D classification
see: http://link.springer.com/chapter/10.1007%2F978-3-642-25367-6_32#page-1
"""
def noisyRGBToBinary(img):
    imgW, imgH = img.size[0], img.size[1]
    mat1D = [[0 for i in range(imgW)] for j in range(imgH)]
    for x in range(imgW):
        for y in range(imgH):
            mat1D[y][x] = int(toGray(img.getpixel((x, y))))
    #range of grayscale values
    L = 256
    #pixel value, mean of neighbors, median of neighbors
    dim = 3
    #how far from current pixel to consider for mean and median
    neighborRange = 1
    mat3D = [[[0 for d in range(dim)] for i in range(imgW)] for j in range(imgH)]
    histograms = [[0 for i in range(L)] for j in range(dim)]
    for x in range(imgW):
        for y in range(imgH):
            adjVals = []
            for yc in range(y - neighborRange, y + neighborRange + 1):
                for xc in range(x - neighborRange, x + neighborRange + 1):
                    if yc > -1 and yc < imgH and xc > -1 and xc < imgW and not(yc == y and xc == x):
                        adjVals.append(mat1D[yc][xc])
            adjVals = np.array(adjVals)
            pixVal = mat1D[y][x]
            meanAdj = int(np.mean(adjVals))
            medAdj = int(np.median(adjVals))
            mat3D[y][x] = [pixVal, meanAdj, medAdj]
            histograms[0][pixVal] += 1
            histograms[1][meanAdj] += 1
            histograms[2][medAdj] += 1
    threshholds = []
    for i in range(dim):
        threshholds.append(otsu(histograms[i], L, imgW * imgH))
    for x in range(imgW):
        for y in range(imgH):
            vals = [mat3D[y][x][d] for d in range(dim)]
            #point = threshhold, slope = -1
            f = lambda p: sum([p[i] - threshholds[i] for i in range(dim)])
            mat1D[y][x] = L - 1 if f(vals) >= 0 else 0
    #matrix is binary image
    matDim1 = numpify2DList(mat1D)
    matDim2, matDim3 = np.copy(matDim1), np.copy(matDim1)
    matRGB = np.stack((matDim1, matDim2, matDim3), axis = -1)
    return Image.fromarray(np.uint8(matRGB))
imgName = "kinect2"
imgExt = "png"
img = Image.open(imgName + "." + imgExt)
print(img.getpixel((0, 0)))
iOut = noisyRGBToBinary(img)
iOut.save(imgName + "Out." + imgExt)
