import numpy as np
import helpFunctions as hpf

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
		return ("Time: "+ timeString +" ,\t"+repr(self.position) + ",\t" + repr(self.velocity))
	
	__repr__ = __str__

	def state(self):
		from numpy import array
		return array([self.position.x, self.position.y, self.velocity.x, self.velocity.y])

class Target:
	def __init__(self, initialTarget, scanIndex,measurementNumber,measurement,
				 initialState, initialStateCovariance, Parent = None, cummulativeNLLR = 0.0, residualCovariance = None):
		from pykalman import KalmanFilter
		self.isAlive = True
		self.parent = Parent
		self.initial = initialTarget
		self.scanIndex = scanIndex
		self.measurementNumber = measurementNumber
		self.measurement = measurement
		self.cummulativeNLLR = cummulativeNLLR
		self.trackHypotheses = [] #children in the track hypothesis tree
		self.filteredStateMean = initialState
		self.filteredStateCovariance = initialStateCovariance
		self.predictedStateMean = None
		self.predictedStateCovariance = None
		self.residualCovariance = residualCovariance

	def __repr__(self):
		from time import ctime
		if self.predictedStateMean is not None:
			predStateStr = " \tPredState: " 	+ str(self.predictedStateMean)
		else:
			predStateStr = ""
		return (repr(self.initial) 
				+ " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR)
				+ predStateStr
				)

	def __str__(self, level=0, hypIndex = 0):
		ret = ""
		if level != 0:
			ret += "\t" + "\t"*level + "H" + str(hypIndex)+":\t" +repr(self)+"\n"
		for hypIndex, hyp in enumerate(self.trackHypotheses):
			ret += hyp.__str__(level+1, hypIndex)
		return ret


	def depth(self, count = 0):
		if len(self.trackHypotheses):
			return self.trackHypotheses[0].depth(count +1)
		return count

	def kfPredict(self, A, Q, b, Gamma):
		import pykalman.standard as pk
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(A,Gamma.dot(Q.dot(Gamma.T)),b,self.filteredStateMean,self.filteredStateCovariance))

	def associateMeasurements(self, measurementList, sigma, P_d, lambda_ex, scanIndex, C, R, d, usedMeasurements):
		from pykalman.standard import _filter_correct
		time = measurementList.time
		measurements = measurementList.measurements
		predictedPosition = Position(self.predictedStateMean[0], self.predictedStateMean[1])
		zeroInitTarget = initialTarget(predictedPosition, self.initial.velocity, time)
		nllr = hpf.NLLR(0, P_d)
		self.trackHypotheses.append( Target(zeroInitTarget, scanIndex, 0, predictedPosition, 
							zeroInitTarget.state(), self.predictedStateCovariance, self, self.cummulativeNLLR + nllr)) #Zero-hypothesis
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma, C, R):
				(_, state, filtCov, resCov) = _filter_correct(
					C, R, d, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				filteredPosition = Position(state[0], state[1])
				deltaPosArray = state[0:2] - self.initial.position.toarray()
				deltaTime = time - self.initial.time
				velocityArray = deltaPosArray / deltaTime
				velocity = Velocity(velocityArray[0], velocityArray[1])
				virtualInitialTarget = initialTarget(filteredPosition,velocity, time)
				predictedMeasurement = np.dot(C,self.predictedStateMean)
				nllr = hpf.NLLR(None,P_d, measurement, predictedMeasurement, lambda_ex, resCov)
				self.trackHypotheses.append( Target(virtualInitialTarget, scanIndex, measurementIndex+1, 
											measurement,state, filtCov, self, self.cummulativeNLLR + nllr, resCov) )
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
