
from PyControl import *
import numpy as np
import time
pi = 3.14159


r = PyControl("/dev/ttyACM0")#"COM6",115200, .04, [.505, .2601, .234-.0043, 0.0164], [.12, 0, -.08])
#
#print r.getState()
# a = r.getState()
speed = 40*pi/180.0
r.traj(0,3,speed)
#time.sleep(1)
#r.sendStateRequest([1,2,3,4,5,6])
#time.sleep(.5)
#print r.getTarget()

#for i in range(0,100):
#	print r.getState()
#	r.sendStateRequest([1,2,3,4,5,6])
#	print r.getTarget()


r.traj(0,4,speed)
#r.sendControls([30,0,0,0,0,0])
#time.sleep(.8)
#time.sleep(.8)

#time.sleep(2)
r.stop()
r.ser.close()

