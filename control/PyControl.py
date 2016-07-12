
import time
import serial
import numpy as np
import math as m


class PyControl:
    def __init__(self, comm = "/dev/ttyACM0",baudrate=2000000,timeout=.15):
        # initialize Serial Connection
        self.ser = serial.Serial(comm,baudrate)
        self.ser.setTimeout(timeout)
        time.sleep(1)
        self.timeDelay = .03
        self.numStates = 6
          
        
    def stop(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("u")
        return
    
    def sendStateRequest(self,requests):
        # print 1
        self.ser.flushOutput()
        #print 2
        self.ser.flushInput()
        #print 3
        self.ser.write("a")
        #print 4
        #print "requests, ", requests
        for i in range(0,self.numStates):
            #print i+self.numStates
            val = int(requests[i]*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))
        return
        
    def sendControls(self, requests):
        self.ser.flushOutput()
        PWMs = []
        for req in requests:
            if req >= 0:
                PWMs.append(int(req))
                PWMs.append(0)
            else:
                PWMs.append(0)
                PWMs.append(int(abs(req)))
        self.ser.write("s")
        for e in PWMs:
            self.ser.write(chr(e))
            
    def traj(self,stage,target,speed):
        currState = self.getState()
        partitions = int(abs((target-currState[stage])/(speed*(self.timeDelay))))
        change = (target-currState[stage])/partitions
        targetState = currState
        
        for i in range(0,partitions):
            #print "1, ", self.getState()
            time.sleep(self.timeDelay)
            #print "2, ", self.getState()
            targetState[stage] += change
            #print targetState
            self.sendStateRequest(targetState)
            # print "3, ", self.getState()
            
        for i in range(0,self.numStates):
            time.sleep(self.timeDelay)
            self.sendStateRequest(targetState)
            
        time.sleep(self.timeDelay)
        time.sleep(.4)
        print 'Error: '
        print (targetState[stage]-self.getState()[stage])
        
        return


    def getState(self):
        # Returns Array: Rotation, Elevation,Extension,Wrist,Jaws,Turntable  
        
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,self.numStates):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
        return sensorVals
        
    def getTarget(self):
        # Returns Array: Rotation, Elevation,Extension,Wrist,Jaws,Turntable
        self.ser.flushInput()
        self.ser.write("t")
        sensorVals = []
        for i in range(0,self.numStates):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
        return sensorVals
        
    def getOutput(self):
        # Returns Array: Rotation, Elevation,Extension,Wrist,Jaws,Turntable
        self.ser.flushInput()
        self.ser.write("o")
        sensorVals = []
        for i in range(0,self.numStates):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
        return sensorVals
        
        
    def getPots(self):
        self.ser.flushInput()
        self.ser.write("p")
        sensorVals = []
        for i in range(0,self.numStates):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'  
                
        return sensorVals
        
    def getCurrents(self):
        self.ser.flushInput()
        self.ser.write("c")
        sensorVals = []
        for i in range(0,4):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'   
        return sensorVals
        
    def getEncoders(self):
        self.ser.flushInput()
        self.ser.write("e")
        sensorVals = []
        for i in range(0,3):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'   
        return sensorVals
    
        
        
    
        
    
    

    
        
