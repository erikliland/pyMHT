<<<<<<< HEAD
import numpy as np

class Position:
	def __init__(self,*args,**kwargs):
		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None)  and (y is not None):
			self.x = x
			self.y = y
		elif len(args) == 1:
			self.x = args[0][0]
			self.y = args[0][1]
		elif len(args) == 2:
			self.x = float(args[0])
			self.y = float(args[1])
		else:
			raise ValueError("Invalid arguments to Position")

	def __add__(self,other):
		return Position(self.x+other.x, self.y + other.y)
	
	def __str__(self):
		return 'Pos: ({: 9.2f},{: 9.2f})'.format(self.x, self.y)
	
	def __repr__(self):
		return '({:.3e},{:.3e})'.format(self.x, self.y)
	
	def toarray(self):
		return np.array([self.x,self.y])

	def __sub__(self,other):
		return Position(self.x-other.x, self.y-other.y)

	def __mul__(self, other):
		return Position(self.x * other, self.y * other)

	def __div__(self, other):
		return Position(self.x / other, self.y / other)

class Velocity:
	def __init__(self,*args,**kwargs):
		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None) and (y is not None):
			self.x = x
			self.y = y
		elif len(args) == 1:
			self.x = args[0][0]
			self.y = args[0][1]
		elif len(args) == 2:
			self.x = args[0]
			self.y = args[1]
		else:
			raise ValueError("Invalid arguments to Position")

	def __str__(self):
		return "("+'{: 6.2f}'.format(self.x)+","+'{: 6.2f}'.format(self.y)+")"

	def __repr__(self):
		return "Vel: " + str(self)

	def __mul__(self,other):
		return Velocity(self.x * other, self.y * other)

	def __div__(self,other):
		return Velocity(self.x / other, self.y / other)

	def toarray(self):
		return np.array([self.x,self.y])

class MeasurementList:
	def __init__(self, Time, measurements = []):
		self.time = Time
		self.measurements = measurements

	def add(self, measurement):
		self.measurements.append(measurement)

	def __str__(self):
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return "Time: "+timeString+"\tMeasurements:\t"+repr(self.measurements)

	__repr__ = __str__

=======
import numpy as np

class Position:
	def __init__(self,*args,**kwargs):
		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None)  and (y is not None):
			self.x = x
			self.y = y
		elif len(args) == 1:
			self.x = args[0][0]
			self.y = args[0][1]
		elif len(args) == 2:
			self.x = float(args[0])
			self.y = float(args[1])
		else:
			raise ValueError("Invalid arguments to Position")

	def __add__(self,other):
		return Position(self.x+other.x, self.y + other.y)
	
	def __str__(self):
		return 'Pos: ({: 9.2f},{: 9.2f})'.format(self.x, self.y)
	
	def __repr__(self):
		return '({:.3e},{:.3e})'.format(self.x, self.y)
	
	def toarray(self):
		return np.array([self.x,self.y])

	def __sub__(self,other):
		return Position(self.x-other.x, self.y-other.y)

	def __mul__(self, other):
		return Position(self.x * other, self.y * other)

	def __div__(self, other):
		return Position(self.x / other, self.y / other)

class Velocity:
	def __init__(self,*args,**kwargs):
		x = kwargs.get('x')
		y = kwargs.get('y')
		if (x is not None) and (y is not None):
			self.x = x
			self.y = y
		elif len(args) == 1:
			self.x = args[0][0]
			self.y = args[0][1]
		elif len(args) == 2:
			self.x = args[0]
			self.y = args[1]
		else:
			raise ValueError("Invalid arguments to Position")

	def __str__(self):
		return "("+'{: 6.2f}'.format(self.x)+","+'{: 6.2f}'.format(self.y)+")"

	def __repr__(self):
		return "Vel: " + str(self)

	def __mul__(self,other):
		return Velocity(self.x * other, self.y * other)

	def __div__(self,other):
		return Velocity(self.x / other, self.y / other)

	def toarray(self):
		return np.array([self.x,self.y])

class MeasurementList:
	def __init__(self, Time, measurements = []):
		self.time = Time
		self.measurements = measurements

	def add(self, measurement):
		self.measurements.append(measurement)

	def __str__(self):
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return "Time: "+timeString+"\tMeasurements:\t"+repr(self.measurements)

	__repr__ = __str__

>>>>>>> bdc854266624b31dcf021d5ed28f1a279df23d82
