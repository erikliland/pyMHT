#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class Position:
	def __init__(self,*args,**kwargs):

		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None)  and (y is not None):
			self.position = np.array([x,y])
		elif len(args) == 1:
			self.position = np.array(args[0])
		elif len(args) == 2:
			self.position = np.array([args[0],args[1]])
		else:
			raise ValueError("Invalid arguments to Position")

	def x(self):
		return self.position[0]

	def y(self):
		return self.position[1]

	def __str__(self):
		return 'Pos: ({0: .2f},{1: .2f})'.format(self.position[0],self.position[1])
	
	def __repr__(self):
		return '({0:.3e},{1:.3e})'.format(self.position)
	
	def toarray(self):
		return self.position

	def __add__(self,other):
		return Position(np.add(self.position,other.position))

	def __sub__(self,other):
		return Position(np.subtract(self.position,other.position))

	def __mul__(self, other):
		return Position(np.multiple(self.position,other))

	def __div__(self, other):
		return Position(np.divide(self.position, other))

	def plot(self, measurementNumber, scanNumber = None, **kwargs):
		if measurementNumber == 0:
			plt.plot(self.position[0],self.position[1],color = "black",fillstyle = "none", marker = "o")
		else:
			plt.plot(self.position[0], self.position[1],'kx')
		if (scanNumber is not None) and (measurementNumber is not None) and kwargs.get("labels",False):
			ax = plt.subplot(111)
			ax.text(self.position[0], self.position[1],str(scanNumber)+":"+str(measurementNumber), size = 7, ha = "left", va = "top") 

class Velocity:
	def __init__(self,*args,**kwargs):
		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None) and (y is not None):
			self.position[0] = x
			self.position[1] = y
		elif len(args) == 1:
			self.position[0] = args[0][0]
			self.position[1] = args[0][1]
		elif len(args) == 2:
			self.position[0] = args[0]
			self.position[1] = args[1]
		else:
			raise ValueError("Invalid arguments to Velocity")

	def __str__(self):
		return 'Vel: ({: .2f},{: .2f})'.format(self.position[0], self.position[1])

	def __repr__(self):
		return '({:.3e},{:.3e})'.format(self.position[0], self.position[1])

	def __mul__(self,other):
		return Velocity(self.position[0] * other, self.position[1] * other)

	def __div__(self,other):
		return Velocity(self.position[0] / other, self.position[1] / other)

	def toarray(self):
		return np.array([self.position[0],self.position[1]])

class MeasurementList:
	def __init__(self, Time, measurements = []):
		self.time = Time
		self.measurements = measurements

	def add(self, measurement):
		self.measurements.append(measurement)

	def __str__(self):
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return "Time: "+timeString+"\tMeasurements:\t"+"".join([str(measurement) for measurement in self.measurements])

	__repr__ = __str__

	def plot(self, **kwargs):
		for measurementIndex, measurement in enumerate(self.measurements):
			measurement.plot(measurementIndex+1,**kwargs)