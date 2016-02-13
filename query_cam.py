from multiprocessing import Process, Queue
from time import sleep, time
from Logger import Logger
from pipeline.bincam import BinaryCamera
from threading import Lock


class query_cam(Process):

	def __init__(self, cam):
		'''
			Gets the camera to be queried so that its queue will be empty
		'''
		Process.__init__(self)
		self.frame = None
		self.bc = cam
		self.running = False
		self.lock = Lock()

	def run(self):
		self.running = True
		while(self.running):
			self.frame = self.bc.read_frame()
			# print self.frame


	# def start(self):
	# 	Process.start(self)
	# 	# self.frame = self.bc.read_frame()


	def read_frame(self):
		return self.bc.read_frame()