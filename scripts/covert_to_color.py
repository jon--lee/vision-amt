from options import AMTOptions
import random
import cv2
import IPython
import numpy as np
import numpy.linalg as LA

def compileColorDir(): 
    path_o = AMTOptions.originals_dir
    path_c = AMTOptions.colors_dir
    deltas_path = AMTOptions.deltas_file

    deltas_file = open(deltas_path, 'r')

    for line in deltas_file:            
        labels = line.split()
        img_name = path_o+labels[0]
        img = cv2.imread(img_name,1)
        # cv2.imshow("image",img)
        # cv2.waitKey(30)
        print path_c+labels[0]
        img = cv2.resize(img.copy(), (250, 250))
        cv2.imwrite(path_c+labels[0],img)


if __name__ == '__main__':

    compileColorDir()

