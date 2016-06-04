from helper import Rectangle, sign, cosine, sine, w, h, getRot
from PIL import Image
import numpy as np
import math
import random

# Program parameters
# number of images to output
numArrangements = 30
# increase to make goal positions closer to center
# do not decrease below 1.0
goalVariance = 2.0

if (goalVariance < 1.0):
    goalVariance = 1.0
for k in range(0, numArrangements):
    images = [1, 2, 3, 4]
    obstacles = []
    background = Image.open("back.png")
    goal = Image.open("goal.png")
    # radius for positioning UL corner of goal
    radius = 155
    # radius for checking if objects are inside circle
    trueRadius = 185
    # x from 0 to 310; y from 155 - sqrt(-(x-310)x) to 155 + sqrt(-(x-310)x)
    x = int(np.random.normal(radius, (radius/goalVariance)/3))
    yRadius = math.sqrt(-1 * x * (x - radius * 2))
    y = int(np.random.normal(radius, (yRadius)/3))
    gObj = Rectangle(x, goal.size[0], y, goal.size[1])
    obstacles.append(gObj)
    background.paste(goal, (x, y), goal.convert('RGBA'))
    # starting angle relative to goal
    theta = random.randrange(0, 360)
    # angle increment between objects
    thetaInc = 90
    # angle increment when placing one object
    thetaShift = 5
    for i in range(1, 5):
        # tracks length of current iteration
        currentTheta = 0
        # selects objects in random order
        j = random.randrange(0, len(images))
        name = "obj" + str(images[j]) + ".png"
        images.pop(j)
        obj = Image.open(name)
        # random small rotation of object
        rotLimit = 40
        rObj = obj.rotate(getRot(rotLimit), Image.NEAREST, True)
        foundLoc = False
        reRotate = False
        # iterates to find valid placement for current object
        while (not foundLoc):
            # increases rotation of object if it did not fit
            if (reRotate):
                rotLimit += 20
                rObj = obj.rotate(getRot(rotLimit), Image.NEAREST, True)
                reRotate = False
            # avoid infinite loops
            if (rotLimit > 700):
                foundLoc = True
            hasSpace = True
            radialD = w(goal)/2
            # iterates distance from goal to find valid placement at current angle
            while (not foundLoc and hasSpace):
                objx = int(gObj.cenX + radialD * cosine(theta))
                objy = int(gObj.cenY + radialD * sine(theta))
                z = Rectangle(objx, w(rObj), objy, h(rObj))
                foundLoc = not z.hasCollision(obstacles)
                hasSpace = z.insideCircle(trueRadius, trueRadius, trueRadius)
                foundLoc = foundLoc and hasSpace
                radialD += 5
                if (foundLoc):
                    background.paste(rObj, (objx, objy), rObj.convert('RGBA'))
                    obstacles.append(z)
                # change angle if no valid placement is available at current angle
                elif (not hasSpace):
                    currentTheta += thetaShift
                    theta += thetaShift
                # changes object rotation if no valid placement is available at any angle
                if (currentTheta > 720):
                    currentTheta = 0
                    reRotate = True
        theta += thetaInc
    # save arrangements as numbered png files
    fileout = str(k) + ".png"
    background.save(fileout)
