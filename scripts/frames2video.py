"""
Given a list of directory names in record_frames dir,
create videos to store in record_videos dir with specified iteration

Specify the directories of the frames and the category where the final
video will end up (e.g. net3_iteration1)
"""

import cv2
import os
import numpy as np
from options import Options


def num_from_filename(filename):
    return int(filename.split('frame_', 1)[1][:-4])

dirs = ['recording_01-14-2016_16h10m37s']
category = "net2_iteration2"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for d in dirs:
    
    print "Compiling '" + d + "' to video file..."

    in_path = Options.frames_dir + d + "/"
    category_path = Options.videos_dir + category + '/'
    if not os.path.exists(category_path): os.makedirs(category_path)
    out_path = Options.videos_dir + category + "/" + d + ".mov"
    writer = cv2.VideoWriter(out_path, fourcc, 10.0, (Options.WIDTH,Options.HEIGHT))
    
    filenames = [ in_path + fn for fn in os.listdir(in_path) if fn.endswith('.jpg') ]
    filenames = sorted(filenames, key=num_from_filename)

    for fn in filenames:
        im = cv2.imread(fn)
        writer.write(im)

print "Done."
