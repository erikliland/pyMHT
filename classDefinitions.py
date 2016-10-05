import numpy as np

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
	def __init__(self, initialTarget, scanIndex,measurementNumber,measurement,
				 initialState, initialStateCovariance, residualCovariance = None):
		from pykalman import KalmanFilter
		self.isAlive = True
		self.initial = initialTarget
		self.scanIndex = scanIndex
		self.measurementNumber = measurementNumber
		self.measurement = measurement
		self.cummulativeNLLR = 0.0
		self.trackHypotheses = [] #children in the track hypothesis tree
		self.filteredStateMean = initialState
		self.filteredStateCovariance = initialStateCovariance
		self.predictedStateMean = None
		self.predictedStateCovariance = None
		self.residualCovariance = residualCovariance

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


	def depth(self, count = 0):
		if len(self.trackHypotheses):
			return self.trackHypotheses[0].depth(count +1)
		return count

	def kfPredict(self, A, Q, b, Gamma):
		import pykalman.standard as pk
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(A,Gamma.dot(Q.dot(Gamma.T)),b,self.filteredStateMean,self.filteredStateCovariance))

	def associateMeasurements(self, measurementList, sigma, scanIndex, C, R, d, usedMeasurements):
		useFilteredEstimate = True
		from pykalman.standard import _filter_correct
		time = measurementList.time
		measurements = measurementList.measurements
		predictedPosition = Position(self.predictedStateMean[0], self.predictedStateMean[1])
		zeroInitTarget = initialTarget(predictedPosition, self.initial.velocity, time)
		self.trackHypotheses.append( Target(zeroInitTarget, scanIndex, 0, predictedPosition, zeroInitTarget.state(), self.predictedStateCovariance)) #Zero-hypothesis
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma, C, R):
				(_, state, filtCov, resCov) = _filter_correct(
					C, R, d, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				if useFilteredEstimate:
					filteredPosition = Position(state[0], state[1])
				else:
					filteredPosition = measurement
				deltaPosArray = filteredPosition.toarray() - self.initial.position.toarray()
				deltaTime = time - self.initial.time
				velocityArray = deltaPosArray / deltaTime
				velocity = Velocity(velocityArray[0], velocityArray[1])
				virtualInitialTarget = initialTarget(filteredPosition,velocity, time)
				self.trackHypotheses.append( Target(virtualInitialTarget, scanIndex, measurementIndex+1, measurement,state, filtCov, resCov) )
				usedMeasurements[measurementIndex] = True

	def measurementIsInsideErrorEllipse(self,measurement, sigma, C, R):
		from numpy import cos, sin, power
		B = C.dot(self.predictedStateCovariance.dot(C.T))+R
		lambda_, v = np.linalg.eig(B)
		deltaX = measurement.x - self.predictedStateMean[0]
		deltaY = measurement.y - self.predictedStateMean[1]
		a = np.sqrt(lambda_[0])*sigma
		b = np.sqrt(lambda_[1])*sigma
		angle = np.rad2deg( np.arctan2( lambda_[1], lambda_[0]) )
		return power( cos(angle)*deltaX+sin(angle)*deltaY,2 )/power(a,2)  + power( sin(angle)*deltaX-cos(angle)*deltaY,2)/power(b,2) <= 1
		#http://stackoverflow.com/questions/7946187/point-and-ellipse-rotated-position-test-algorithm

	def predictedPosition(self):
		return Position(self.predictedStateMean[0], self.predictedStateMean[1])

def reprHypothesis(target, list):
	if len(target.trackHypotheses == 0):
		return list

	return list.append(reprHypothesis(target, list))
