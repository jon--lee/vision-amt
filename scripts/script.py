import cv2
import os
import numpy as np

path = "deploy3"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(path + '.mov', fourcc, 10.0, (420,420))



nums = []
for filename in os.listdir(path + '/'):
    if filename.endswith('.jpg'):
        _, rest = filename.split('_')
        num, _ = rest.split('.')
        num = int(num)
        nums.append(num)

nums.sort()
print nums
for num in nums:
    filename = path + "/frame_" + str(num) + ".jpg"
    im = cv2.imread(filename)
    writer.write(im)
    
