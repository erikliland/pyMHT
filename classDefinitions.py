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

	def __sub__(self,other):
		return Position(self.x-other.x, self.y-other.y)

	def __mul__(self, other):
		return Position(self.x * other, self.y * other)

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
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return ("Time: "+ timeString +" \t"+repr(self.position) + ",\t" + repr(self.velocity))
	
	__repr__ = __str__

	def state(self):
		from numpy import array
		return array([self.position.x, self.position.y, self.velocity.x, self.velocity.y])

class Target:
	def __init__(self, initialTarget, scanIndex, kfData):
		from pykalman import KalmanFilter
		self.initial = initialTarget
		self.scanIndex = scanIndex
		self.isAlive = True
		self.trackHypothesis = [] ##children in the track hypothesis tree
		self.filteredStateMean = kfData.initialStateMean
		self.filteredStateCovariance = kfData.initialStateCovariance
		self.predictedStateMean = None
		self.predictedStateCovariance = None

	def __str__(self):
		from time import ctime
		return (repr(self.initial) 
			+ " \tScanIndex: " 	+ str(self.scanIndex) 
			+ " \tIsAlive: " 	+ str(self.isAlive) 
			+ " \t#Hypothesis: "+ str(len(self.trackHypothesis))
			+ " \tPredState: " 	+ str(self.predictedStateMean)
			# + " \tPredCov: " 	+ repr(self.predictedStateCovariance)
			+ " \tFiltState: " 	+ str(self.filteredStateMean)
			# + " \nFiltCov: " 	+ str(self.filteredStateCovariance)
			)

	__repr__ = __str__

	def assosiateMeasurements(self, measurements, sigma, scanIndex, time):
		from tomht import getKalmanFilterInitData
		foundMeasurement = False
		measurementCandidates = []
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma):
				foundMeasurement = True
				deltaPosArray = measurement.array() - self.initial.position.array()
				deltaTime = time - self.initial.time
				velocityArray = deltaPosArray / deltaTime
				velocity = Velocity(velocityArray[0], velocityArray[1])
				virtualInitialTarget = initialTarget(measurement,velocity, time)
				kfData = getKalmanFilterInitData(virtualInitialTarget)
				self.trackHypothesis.append( Target(virtualInitialTarget, scanIndex, kfData) )
		if not foundMeasurement:
			measurementCandidates.append(-1)
			self.trackHypothesis.append( Target() )

	def measurementIsInsideErrorEllipse(self,measurement, sigma):
		from numpy.linalg import eig
		from numpy import sqrt, rad2deg, arccos, cos, sin, power
		lambda_, v = eig(self.predictedStateCovariance[0:2,0:2])
		lambda_ = sqrt(lambda_)
		deltaX = measurement.x - self.predictedStateMean[0]
		deltaY = measurement.y - self.predictedStateMean[1]
		a = lambda_[0]*sigma
		b = lambda_[1]*sigma
		angle = arccos(v[0,0])
		sum = power( cos(angle)*deltaX+sin(angle)*deltaY,2 )/power(a,2)  + power( sin(angle)*deltaX-cos(angle)*deltaY,2)/power(b,2)
		if sum <= 1:
			return True
		else:
			return False

	def kfPredict(self, A, Q0, b):
		import pykalman.standard as pk
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(A,Q0,b,self.filteredStateMean,self.filteredStateCovariance))


class Track:
	def __init__(self, initialState, time, initialScanIndex):
		from numpy import eye
		self.measurementIndex = None
		self.cummulativeScore = None
		self.parrent = None
		self.trackHypothesis = []
	
	def __str__(self):
		return "Score: " + str(self.cummulativeScore)

	__repr__ = __str__

	
