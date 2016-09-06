class Position:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __add__(self,other):
		return Position(self.x+other.x, self.y + other.y)
	def __str__(self):
		return "Pos: ("+str(round(self.x,2))+","+str(round(self.y,2))+")"
	__repr__ = __str__

	def array(self):
		from numpy import array
		return array([self.x,self.y])

class Velocity:
	def __init__(self, x, y):
		self.x = x #m/s
		self.y = y #m/s
	
	def __str__(self):
		return "Vel: ("+str(round(self.x,2))+","+str(round(self.y,2))+")"
	
	__repr__ = __str__

	def __mul__(self,other):
		return Velocity(self.x * other, self.y * other)

class MeasurementList:
	def __init__(self, Time):
		self.time = Time
		self.measurements = []

	def add(self, measurement):
		self.measurements.append(measurement)

	def __str__(self):
		from time import ctime
		return "Time: "+str(ctime(self.time))+"\tMeasurements:\t"+repr(self.measurements)

	__repr__ = __str__

class initialTarget:
	def __init__(self, pos, vel, time):
		self.position = pos
		self.velocity = vel
		self.time = time

	def __str__(self):
		from time import ctime
		return "Time: "+str(ctime(self.time))+"  \t"+repr(self.position) + ",\t" + repr(self.velocity)
	
	__repr__ = __str__

class Target:
	def __init__(self, initialTarget):
		self.initial = initialTarget
		self

	def __str__(self):
		from time import ctime
		return repr(self.initial)
	
	__repr__ = __str__