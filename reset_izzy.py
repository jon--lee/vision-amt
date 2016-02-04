import sys
sys.path[0] += '/gripper/'

from gripper.TurnTableControl import TurnTableControl
from gripper.PyControl import PyControl
import random
import numpy as np
import numpy.linalg as lin
import time
pi = np.pi



class reset():

	def __init__(self, izzy_ctrl, turn_ctrl):
		self.izzy = izzy_ctrl
		self.turn = turn_ctrl

	def reset_move(self):
		val = random.random()

		# set the arm to the center, retracted
		base_state = [pi+.46, .00772, .0082, 5.46418, -.01112, 0]
		time.sleep(3)

		new_state = list(base_state) 
		new_state[1] += .2
		self.izzy.sendStateRequest(new_state)
		time.sleep(3)

		new_state = list(base_state)
		new_state[2] = .312
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[0] += .6
		# zeke.gotoState(ZekeState([None, None, .3, None, None, None]))
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[1] = base_state[1]
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[0] -= .5
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[1] += .15
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[0] = base_state[0]
		new_state[0] -= .6
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[1] = base_state[1]
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[0] += .5
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[1] += .15
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		new_state[2] = base_state[2]
		self.izzy.sendStateRequest(new_state)

		time.sleep(3)
		self.izzy.sendStateRequest(base_state)

		time.sleep(3)
		self.turn.sendStateRequest([(.5 - val) * pi])

	def wait_to(self, state):
		current = np.array(self.izzy.getState())
		nxt = np.array(state)
		return lin.norm(current - nxt)
		
		
turn = TurnTableControl()
izzy = PyControl()
res = reset(izzy, turn)
