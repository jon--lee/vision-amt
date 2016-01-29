"""
Script that aggregates datasets: compiles "trian.txt" and "test.txt" with image paths with scaled controls.
images will always be renamed to "<<dataset name>>_img_<<img number>>.jpg" 
If specified, segments and moves those images to images/ folder.

Run:
    # segment and import images and compile trian/test sets
    python ./scripts/dagger.py -d "dataset1 dataset2" --segment
    
    # only compile train/test sets
    python ./scripts/dagger.py -d "dataset1 dataset2"

"""
from options import Options
from dataset import Dataset
from pipeline import bincam
from Net import hdf
import cv2
import random
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--datasets', required=True)
ap.add_argument('-s', '--segment', required=False, action='store_true')
args = vars(ap.parse_args())

segment = args['segment']
dataset_names = args['datasets'].split(' ')


# erase former data in files and open new writers
open("Net/hdf/train.txt", 'w').close()
open("Net/hdf/test.txt", 'w').close()
train_writer = open("Net/hdf/train.txt", "w+")
test_writer = open("Net/hdf/test.txt", "w+")


for dataset_name in dataset_names:
    print "processing " + dataset_name + "..."
    ds = Dataset.get_ds(Options(), dataset_name)
    controls_reader = open(ds.path + 'controls.txt', 'r')
    bc = bincam.BinaryCamera(ds.path + 'meta.txt')

    for line in controls_reader:
        split = line.split(' ')
        filename = split[0]
        new_filename = ds.name + '_' + filename
        
        controls = [ float(x)/float(s) + float(t) for x, s, t in 
                zip(split[1:], Options.scales, Options.translations) ]
        
        # should segment and import images?
        if segment:
            im = cv2.imread(ds.path + filename)
            im = bc.pipe(im)
            cv2.imwrite(Options.data_dir + "images/" + new_filename, im)

        controls_string = Dataset.controls2str(controls)
        line = Options.data_dir + "images/" + new_filename + controls_string + "\n"
        
        if random.random() > .2:    train_writer.write(line)
        else:                       test_writer.write(line)

train_writer.close()
test_writer.close()

# load images into h5 files
print "converting train set to hdf..."
hdf.img2hdf('Net/hdf/train', Options.hdf_dir + "train.h5")

print "converting test set to hdf..."
hdf.img2hdf('Net/hdf/test', Options.hdf_dir + "test.h5")

print "Done."
