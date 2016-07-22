import sys
sys.path.append('/home/annal/Izzy/vision_amt/scripts/objects/')
from helper import Rectangle, sign, cosine, sine, w, h, getRot, pasteOn
from PIL import Image
import numpy as np
import math
import random
from pipeline.bincam import BinaryCamera
import sys, os, time, cv2
from Net.tensor import inputdata
from options import AMTOptions

# Program parameters
# number of images to output
numArrangements = 1
# increase to make object cluster centered closer to center of circle
# do not decrease below 1.0
lowGV = 10.0
highGV = 1.0
goalVariance = lowGV
# set variance of which objects to use
lowSD = 0
highSD = 5
stddev = lowSD
# set background to use
transpBack = "back"
actualBack = "back1"
backName = transpBack
# probabilities of objects using normalized gaussian values
# index 5 most common, then 4 and 6, etc.
mean = 5
def pd(x, var, mean):
    res = 1/(math.sqrt(2*math.pi*(var)**2))
    res = res * (math.e)**(-(x-mean)**2/(2*var**2))
    return res
objP = []
total = 0
if stddev == 0:
    total = 4.0
    objP = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
else:    
    for i in range(1, 9):
        val = pd(i, stddev, mean)
        total += val
        objP.append(val)
#normalizing
objP = [x/total for x in objP]

imageNames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
"""Changes white pixels to transparent ones"""
def makeTransparent(model):
    model = model.convert('RGBA')
    datas = model.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    model.putdata(newData)
    return model
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
    gripper = Rectangle(295, 125, 120, 125)
    obstacles = [gripper]
    # grip = Image.new('RGB', (125,125), color=400)
    # white background
    clusterBack = Image.open("scripts/objects/back.png")
    # background of result
    background = Image.open("scripts/objects/" + backName + ".png")
    # radius for checking if objects are inside circle and positioning center
    radius = 180 # 180
    xCenter = 160
    yRadius = 210
    yCenter = 210
    # x from 0 to 310; y from 155 - sqrt(-(x-310)x) to 155 + sqrt(-(x-310)x)
    x = np.random.normal(xCenter, (radius*1.2))
    # print (radius/2), yRadius/2
    # field = Image.new('RGB', (int(radius*1.2),int(yRadius*1.2)), color=400)
    # yRadius = math.sqrt(-1 * x * (x - radius * 2))
    y = np.random.normal(yCenter, (yRadius*1.2))
    center = (x,y)
    order = []
    for i in range(1, 5):
        # selects objects in random order
        j = random.randrange(0, len(images))
        name = "scripts/objects/" + "obj" + str(images[j]) + ".png"
        order.append(images[j])
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
        # foundLoc = (not z.hasCollision(obstacles)) and z.insideCircle(radius, radius, radius)
        foundLoc = (not z.hasCollision(obstacles) and z.insideRectangle((0, 2*yRadius), (2*radius,0)))
        if not foundLoc:
            # print "Failed: ", (x,y)
            return None, None, order
        else:
            pasteOn(clusterBack, obj, objX, objY)
        obstacles.append(z)
    # Random angle from -30 to 30; noise
    theta = (random.random() * 60) - 30
    clusterBack = clusterBack.rotate(theta)
    clusterBack = makeTransparent(clusterBack)
    # pasteOn(background, field, xCenter - int(radius/2*1.2), yCenter- int(yRadius/2*1.2))
    # pasteOn(background, field, 0, 0)
    pasteOn(background, clusterBack, 0, 0)
    return background, center, order

def generate_template():
    cont = True
    center = (0,0)
    while (cont):
        images = []
        while (len(images) < 4):
            img = imageNames[probIndex(objP)]
            if (img not in images):
                images.append(img)
        model, center, order = makeImg(images)
        if (model is not None):
            cont = False
    # make background transparent
    if (backName == "back"):
        model = makeTransparent(model)
    # save arrangements as numbered png files
    # print order
    fileout = "/home/annal/Izzy/vision_amt/scripts/objects/template.png"
    model.save(fileout)
    return center
# generate_template()


# if (goalVariance < 1.0):
#     goalVariance = 1.0
# for k in range(0, numArrangements):
#     cont = True
#     while (cont):
#         images = []
#         while (len(images) < 4):
#             img = imageNames[probIndex(objP)]
#             if (img not in images):
#                 images.append(img)
#         model = makeImg(images)
#         if (model is not None):
#             cont = False
#     # make background transparent
#     if (backName == "back"):
#         model = makeTransparent(model)
#     # save arrangements as numbered png files
#     fileout = str(k) + ".png"
#     model.save(fileout)

def display_template(bc, template=None):
    if template is None:
        template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")

    template[:,:,1] = template[:,:,2]
    template[:,:,0] = np.zeros((420, 420))
    # template[:,:,2] = np.zeros((420, 420))
    # template = cv2.resize(template, (250, 250))
    while 1:
        frame = bc.read_frame()
        frame = inputdata.im2tensor(frame, channels = 3)
        final = np.abs(-frame + template/255.0)
        cv2.imshow('camera', final)
        a = cv2.waitKey(30)
        if a == 27:
            cv2.destroyAllWindows()
            break
        elif a == ord(' '):
            return 'next'
        time.sleep(.005)

def save_templates(num):
    save_directory = AMTOptions.amt_dir + 'saved_templates/'
    filename = save_directory + 'template_paths.txt'
    paths = open(filename, 'w+')
    for i in range(num):
        center = generate_template()
        name = save_directory + 'template_' + str(i) + '.npy'
        paths.write(name + '\n')
        template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")
        np.save(name, template)
    paths.close()

    
if __name__ == '__main__':
    save_templates(60)


# bc = BinaryCamera('./meta.txt')
# bc.open()
# for i in range(100000):
#     center = generate_template()
#     print center
#     display_template(bc)