import sys
sys.path.append('/usr/lib/python2.7/dist-packages/PIL')
from helper import Rectangle, sign, cosine, sine, w, h, getRot, pasteOn
from PIL import Image
import numpy as np
import math
import random

# Program parameters
# number of images to output
numArrangements = 30
# increase to make object cluster centered closer to center of circle
# do not decrease below 1.0
lowGV = 10.0
highGV = 1.0
goalVariance = lowGV
# set variance of which objects to use
lowSD = 1
highSD = 5
stddev = highSD
# set to true if background should be transparent
whiteBackground = True

# probabilities of objects using normalized gaussian values
# index 5 most common, then 4 and 6, etc.
mean = 5
def pd(x, var, mean):
    res = 1/(math.sqrt(2*math.pi*(var)**2))
    res = res * (math.e)**(-(x-mean)**2/(2*var**2))
    return res
objP = []
total = 0
for i in range(1, 9):
    val = pd(i, stddev, mean)
    total += val
    objP.append(val)
#normalizing
objP = [x/total for x in objP]

imageNames = [1, 2, 3, 4, 5, 6, 7, 8, 9]
"""takes array of probabilities adding to 1 and returns index of chosen event"""
def probIndex(probs):
    choice = random.random()
    curr = 0
    ind = -1
    while (curr < choice):
        ind += 1
        curr += probs[ind]
    return ind

"""Returns image with objects arranged close together"""
def makeImg(images):
    obstacles = []
    background = Image.open("back.png")
    # radius for checking if objects are inside circle and positioning center
    radius = 180
    # x from 0 to 310; y from 155 - sqrt(-(x-310)x) to 155 + sqrt(-(x-310)x)
    x = np.random.normal(radius, (radius/goalVariance)/3)
    yRadius = math.sqrt(-1 * x * (x - radius * 2))
    y = np.random.normal(radius, (yRadius)/3)
    for i in range(1, 5):
        # selects objects in random order
        j = random.randrange(0, len(images))
        name = "obj" + str(images[j]) + ".png"
        images.pop(j)
        obj = Image.open(name)
        objX = x
        objY = y
        # splits objects into four quadrants around point
        if (i % 2 == 0):
            objX -= w(obj)
        if (i > 2):
            objY -= h(obj)
        z = Rectangle(objX, w(obj), objY, h(obj))
        foundLoc = (not z.hasCollision(obstacles)) and z.insideCircle(radius, radius, radius)
        if not foundLoc:
            return None
        else:
            pasteOn(background, obj, objX, objY)
        obstacles.append(z)
    return background

if (goalVariance < 1.0):
    goalVariance = 1.0
for k in range(0, numArrangements):
    cont = True
    while (cont):
        images = []
        while (len(images) < 4):
            img = imageNames[probIndex(objP)]
            if (img not in images):
                images.append(img)
        model = makeImg(images)
        if (model is not None):
            cont = False
    # make background transparent
    if (whiteBackground):
        model = model.convert('RGBA')
        datas = model.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        model.putdata(newData)
    # save arrangements as numbered png files
    fileout = str(k) + ".png"
    model.save(fileout)
