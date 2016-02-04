"""
Script that test runs gripper
Show camera view with "python scripts/test.py -s"
Comment or uncomment "options.record = True"
"""

from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *

from pipeline.bincam import BinaryCamera
from options import Options

import time
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--show', required=False, action='store_true')
args = vars(ap.parse_args())


bincam = BinaryCamera('./meta.txt')
bincam.open()

options = Options()
options.show = args['show']
options.record = False


c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])

while True:
	print c.getUpdates()

