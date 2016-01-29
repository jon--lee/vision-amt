# LFD Grasping in clutter

### Dependencies

* OpenCV/cv2: Ubuntu users can download latest [shell script from here](https://github.com/jayrambhia/Install-OpenCV/tree/master/Ubuntu) and run it. Mac users should follow
 [this tutorial](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/).
* TensorFlow: Install [via pip](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation) is probably easiest.
* [Caffe](http://caffe.berkeleyvision.org/installation.html): Installation page lists Caffe requirements as well.
* h5py: `pip install h5py`
* [pygame](http://www.pygame.org/download.shtml): Only required if you want to control the gripper, but recommended anyway. If you have trouble on mac see [this](http://stackoverflow.com/questions/7775948/no-matching-architecture-in-universal-wrapper-when-importing-pygame).
* If you are using a mac, you may need to install [this driver](http://tattiebogle.net/index.php/ProjectRoot/Xbox360Controller/OsxDriver) version 0.11 for the xbox controller. The most recent version (0.12) does not work with OS X Yosemite.

### Overview

The project can be broken down into five components:
* __Gripper control and data manipulation__ which all takes place in the `vision/` directory. This is the largest component and it provides API to control the gripper, deploy nets, record datasets, and compile to hdf.
* __Caffe nets__ which is in the `vision/net/` dir. Caffe nets are trained using hdf data in the `vision/net/hdf/` dir. Nets are written as prototxt files where the model.prototxt is used for deployment, solver.prototxt for hyperparameters, and trainer.prototxt for training. Train/Test Loss visualization is available. Weights/biases are saved as caffemodels
* __TensorFlow nets__ which is in the `vision/net/tensor/` dir. Nets are all subclasses of TensorNet, which specifies training, deployment, saving, loading. The actual net files just contain the architectures. Variables are saved to ckpts in corresponding folders. Train loss visualization is available.
* __Data__ which is in the `vision/data/` dir. This directory should be mostly untouched. The previous three components should read/write without trouble, although it may be helpful to look through the about.txt and metadata before and after any operations.
* __Options__ which is a single file at `vision/options.py`. This file contains options and info the camera pipeline, nets and gripper interaction. You can create instances and edit appropriately for your setup. Many of the LFD methods depend on instances of Options.

### Instructions and Setup

Be sure to install all the dependencies and ensure they work.

Clone this repo with `git clone https://github.com/mdlaskey/sandbox.git`, cd into it and then `git checkout Vision`. cd into the `vision` directory.

##### Gripper and data instructions

You must determine what USB ports you are using for the gripper and then change the comms appropriately in `gripper/PyControl.py` and `gripper/TurnTable.py`. I don't really understand how all this works. Ask Dave.

If you intend to collect data or deploy a net, you'll need to calibrate the camera. Clear the table and place a small red dot in the center of the turntable. Run `python ./scripts/calibrate.py` from the root directory. A black and white window will appear. The entire turntable should be visible by the camera. Also, the center of the turn table should be marked with a red pixel and one point on the edge of the turntable should be marked with a green pixel. Edit the options.py and repeat until these conditions are met. This process is required for binary segmentation.

There are severals scripts available for testing, deploying, learning, writing to hdf. Run these from the root directory. See descriptions in the headers of each file for more information

* `python scripts/test.py`: Test xbox controller on the gripper. You can see the camera view by adding the `-s` flag. 
* `python scripts/learn.py`: Learns soleley from user actions. A new dataset will be created in `vision/data/datasets/` unless you specify an existing one.
* `python scripts/deploy.py`: Deploys net on the gripper. Specify the net model and caffemodel in the `options` variable. Capable of being overriden by user by holding the "right bumper" of the controller. When user overrides, LFD will behave the same way as learning. Same dataset situation...

You have the option to record from the camera in all of the above scripts. Uncomment or comment out `options.record = True` as you see fit.

* `python scripts/dagger.py -d "dataset0 dataset1 dataset2"`: Aggregates datasets into `vision/data/images/` and writes paths and controls to `vision/net/hdf/train.txt` and `test.txt` with the option to segment images. See file's header for more info. Also writes h5 data for caffe.
* `python scripts/frames2video.py`: Make videos from your frame recordings. See header. Add frame directories to the `dirs` list.


##### Training Nets with Caffe instructions


##### Training Nets with TensorFlow instructions

