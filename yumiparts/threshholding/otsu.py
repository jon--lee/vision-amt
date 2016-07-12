from PIL import Image
import numpy as np

def numpify(arr):
    subs = []
    for sub in arr:
        npSub = np.array(sub)
        subs.append(npSub)
    return np.array(subs)
imgName = "noisy2"
imgExt = "jpg"
img = Image.open(imgName + "." + imgExt)
imgW, imgH = img.size[0], img.size[1]
mat = [[0 for i in range(imgW)] for j in range(imgH)]
toGray = lambda x: 0.3 * x[0] + 0.6 * x[1] + 0.1 * x[2]
histogram = [0 for i in range(256)]
for x in range(imgW):
    for y in range(imgH):
        pix = int(toGray(img.getpixel((x, y))))
        mat[y][x] = pix
        histogram[pix] += 1
"""
https://en.wikipedia.org/wiki/Otsu%27s_method
http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=184351
otsu's algorithm- maximize inter-class variance
splits image along a threshhold value
wB, wF are class probabilities based on number of pixels
mB, mF are class mean
2 threshholds accounts for multiple groupings with same variance
"""
total = imgW * imgH
sum1 = 0
for i in range(0, 256):
        sum1 += i * histogram[i]
sumB, wB, wF = 0, 0, 0
max1, between, thresh1, thresh2 = 0, 0, 0, 0
for i in range(1, 256):
    wB += histogram[i]
    if wB == 0:
        continue
    wF = total - wB
    if wF == 0:
        break
    sumB += i * histogram[i]
    mB = sumB / wB
    mF = (sum1 - sumB) / wF
    between = wB * wF * (mB - mF) * (mB - mF)
    if between >= max1:
        thresh1 = i
        if between > max1:
            thresh2 = i
        max1 = between
thresh = (thresh1 + thresh2)/2
for x in range(imgW):
    for y in range(imgH):
        pix = mat[y][x]
        if pix > thresh:
            pix = 255
        else:
            pix = 0
        mat[y][x] = pix
#matrix is binary image
m = numpify(mat)
m2 = np.copy(m)
m3 = np.copy(m)
rgb = np.stack((m, m2, m3), axis = -1)
iOut = Image.fromarray(np.uint8(rgb))
iOut.save(imgName + "Out." + imgExt)
