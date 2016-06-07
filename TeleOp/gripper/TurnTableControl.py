# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:28:48 2015

@author: David
"""

import time
import serial
import numpy as np
import math as m


class TurnTableControl:
    def __init__(self, comm = "/dev/cu.usbmodem14141",baudrate=115200,timeout=.04, offset = .5):
        # initialize Serial Connection
        self.ser = serial.Serial(comm,baudrate)
        self.ser.setTimeout(timeout)
        time.sleep(1)
        
        self.offset = offset;

        
    def stop(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls([0])
        return
    
    def calibrate(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("c")
        return
        
    def reset(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("r")
        return
        
    def sendStateRequest(self,requests):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for thing in requests:
            val = int(thing*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))

    def sendControls(self,requests):
        # Converts an array of requests to an array of PWM signals sent to the robot
        # Checks out of bounds 
        self.ser.flushOutput()
        PWMs = []
        for i in range(0,len(requests)):
            req = requests[i]
            if req >= 0:
#                if self.state[i+7]>self.maxStates[i]:
#                    req = 0
                PWMs.append(int(req))
                PWMs.append(0)
                
            else:
#                if self.state[i+7]<self.minStates[i]:
#                    req = 0
                PWMs.append(0)
                PWMs.append(int(abs(req)))
        # send PWMs
        for elem in PWMs:
            self.ser.write(chr(elem))
            
        return
        
    def control(self,requests):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls(requests)
        return


    def getState(self):
        # Returns Array: Rotation, Elevation,Extension,Wrist,Jaws,Turntable
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,1):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                time.sleep(.05)
                return 'Comm Failure'
                
   
        return sensorVals
        
    def getPots(self):
        self.ser.flushInput()
        self.ser.write("q")
        sensorVals = []
        for i in range(0,1):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'  
                
        return sensorVals
        
