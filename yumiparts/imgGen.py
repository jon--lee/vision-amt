from stlConvert import generatePose
import math
import random
import time

"""returns probability of gaussian having a specific value"""
def pd(x, var, mean):
    res = 1/(math.sqrt(2*math.pi*(var)**2))
    res = res * (math.e)**(-(x-mean)**2/(2*var**2))
    return res
"""takes array of probabilities adding to 1 and returns index of chosen event"""
def probIndex(probs):
    choice = random.random()
    curr = 0
    ind = -1
    while (curr < choice):
        ind += 1
        curr += probs[ind]
    return ind
imageNames = [1, 2, 3, 4, 5, 6, 7, 8]
mean = 5
stddev = 1
objP = []
total = 0
for i in range(1, 9):
    val = pd(i, stddev, mean)
    total += val
    objP.append(val)
#normalizing
objP = [x/total for x in objP]
images = []
while (len(images) < 4):
    img = imageNames[probIndex(objP)]
    if (img not in images):
        images.append(img)
for i in range(1, 5):
    partNumber = images[i - 1]
    #angle in degrees for rotation about principal axes (30 to - 30)
    # tilt1 = (random.random() * 60) - 30
    # tilt2 = (random.random() * 60) - 30
    tilt1, tilt2 = 0, 0
    #width must be a power of 2- greatly decreases performance when increased
    width = 512
    #s = time.time()
    generatePose(partNumber, tilt1, tilt2, i, width)
    #print("time taken was: " + str(time.time() - s))
