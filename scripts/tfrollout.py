"""
Script that deploys net on gripper
Net is capable of being overridden by xbox controller

Uncomment or comment 'options.record = True' as you see fit.
Specify the deploy model in 'options.model_path' and 'options.weights_path'

"""

from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *

from pipeline.bincam import BinaryCamera
from options import Options
from lfd import LFD

from Net.tensor import net2

bincam = BinaryCamera('./meta.txt')
bincam.open()

options = Options()

t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
izzy = PyControl(115200, .04, [0,0,0,0,0],[0,0,0]); # same with this
c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])

options.tf_net = net2.NetTwo()
options.tf_sess = options.tf_net.load(options.tf_dir + 'net2/net2_01-21-2016_02h14m08s.ckpt')

options.record=True

lfd = LFD(bincam, izzy, t, c, options=options)


dataset_name = "" # if you want to add to existing dataset, specifiy directory name (not path).
                  # else a new one is created in datetime format
lfd.rollout_tf(dataset_name)
