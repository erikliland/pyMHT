class Position:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __add__(self,other):
		return Position(self.x+other.x, self.y + other.y)
	def __str__(self):
		return "Pos: ("+'{: 06.4f}'.format(self.x)+","+'{: 06.4f}'.format(self.y)+")"
	__repr__ = __str__

	def toarray(self):
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
		return "Vel: ("+'{: 06.4f}'.format(self.x)+","+'{: 06.4f}'.format(self.y)+")"

	__repr__ = __str__

	def __mul__(self,other):
		return Velocity(self.x * other, self.y * other)

	def toarray(self):
		from numpy import array
		return array([self.x,self.y])

class MeasurementList:
	def __init__(self, Time):
		self.time = Time
		self.measurements = []

	def add(self, measurement):
		self.measurements.append(measurement)

	def __str__(self):
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return "Time: "+timeString+"\tMeasurements:\t"+repr(self.measurements)

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
	def __init__(self, initialTarget, scanIndex,measurementNumber,measurement, kfData):
		from pykalman import KalmanFilter
		self.isAlive = True
		self.initial = initialTarget
		self.scanIndex = scanIndex
		self.measurementNumber = measurementNumber
		self.measurement = measurement
		self.cummulativeNLLR = 0
		self.trackHypotheses = [] ##children in the track hypothesis tree
		self.filteredStateMean = kfData.initialStateMean
		self.filteredStateCovariance = kfData.initialStateCovariance
		self.predictedStateMean = None
		self.predictedStateCovariance = None

	def __repr__(self):
		from time import ctime
		return (repr(self.initial)
			+ " \tInitialScanIndex: " 	+ str(self.scanIndex) 
			+ " \tIsAlive: " 	+ str(self.isAlive) 
			+ " \t#Hypothesis: "+ str(len(self.trackHypotheses))
			+ " \tNLLR: "		+ str(self.cummulativeNLLR)
			+ " \tPredState: " 	+ str(self.predictedStateMean)
			# + " \tPredCov: " 	+ repr(self.predictedStateCovariance)
			+ " \tFiltState: " 	+ str(self.filteredStateMean)
			# + " \nFiltCov: " 	+ str(self.filteredStateCovariance)
			)

	def __str__(self, level=0):
		ret = ""
		if level != 0:
			ret += str(level)+":" + "\t"*level+repr(str(self.measurementNumber))+"\n"
		for hyp in self.trackHypotheses:
			ret += hyp.__str__(level+1)
		return ret

	def kfPredict(self, A, Q0, b):
		import pykalman.standard as pk
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(A,Q0,b,self.filteredStateMean,self.filteredStateCovariance))

	def associateMeasurements(self, measurementList, sigma, scanIndex, C, R0, d, usedMeasurements):
		from tomht import getKalmanFilterInitData
		from pykalman.standard import _filter_correct
		time = measurementList.time
		measurements = measurementList.measurements
		predictedPosition = Position(self.predictedStateMean[0], self.predictedStateMean[1])
		zeroInitTarget =  initialTarget(predictedPosition, self.initial.velocity, time)  
		kfData = getKalmanFilterInitData(zeroInitTarget)
		self.trackHypotheses.append( Target(zeroInitTarget, scanIndex, 0, predictedPosition, kfData) ) #Zero-hypothesis
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma):
				(K, state, cov) = _filter_correct(
					C, R0, d, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				filteredPosition = Position(state[0], state[1])
				deltaPosArray = filteredPosition.toarray() - self.initial.position.toarray()
				deltaTime = time - self.initial.time
				velocityArray = deltaPosArray / deltaTime
				velocity = Velocity(velocityArray[0], velocityArray[1])
				virtualInitialTarget = initialTarget(filteredPosition,velocity, time)
				kfData = getKalmanFilterInitData(virtualInitialTarget)
				self.trackHypotheses.append( Target(virtualInitialTarget, scanIndex, measurementIndex+1, measurement,kfData) )
				usedMeasurements[measurementIndex] = True

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

	def predictedPosition(self):
		return Position(self.predictedStateMean[0], self.predictedStateMean[1])

def reprHypothesis(target, list):
	if len(target.trackHypotheses == 0):
		return list

	return list.append(reprHypothesis(target, list))
