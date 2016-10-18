import numpy as np
import helpFunctions as hpf
import pykalman.standard as pk

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
			self.x = args[0]
			self.y = args[1]
		else:
			print("Invalid arguments to Position")

	def __add__(self,other):
		return Position(self.x+other.x, self.y + other.y)
	def __str__(self):
		return "("+'{: 06.4f}'.format(self.x)+","+'{: 06.4f}'.format(self.y)+")"
	
	def __repr__(self):
		return "Pos: " + str(self)

	def toarray(self):
		from numpy import array
		return array([self.x,self.y])

	def __sub__(self,other):
		return Position(self.x-other.x, self.y-other.y)

	def __mul__(self, other):
		return Position(self.x * other, self.y * other)

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
			print("Invalid arguments to Position")

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

class InitialTarget:
	def __init__(self, *args, **kwargs):
		pos = kwargs.get('Position')
		vel = kwargs.get('Velocity')
		time= kwargs.get('Time')
		if (pos is not None) and (vel is not None) and (time is not None):
			self.position 	= pos
			self.velocity 	= vel
			self.time 		= time
		elif len(args) == 2:
			self.position 	= Position(args[0][0:2])
			self.velocity 	= Velocity(args[0][2:4])
			self.time 		= args[1]
		elif len(args) == 3:
			self.position 	= args[0]
			self.velocity 	= args[1]
			self.time 		= args[2]
		else:
			print("Invalid arguments to Position")
		

	def __str__(self):
		from time import gmtime, strftime
		timeString = strftime("%H:%M:%S", gmtime(self.time))
		return ("Time: "+ timeString +" ,\t"+repr(self.position) + ",\t" + repr(self.velocity))
	
	__repr__ = __str__

	def state(self):
		from numpy import array
		return array([self.position.x, self.position.y, self.velocity.x, self.velocity.y])

	def plot(self, index = None):
		import matplotlib.pyplot as plt
		plt.plot(self.position.x,self.position.y,"k+")
		if index is not None:
			from numpy.linalg import norm
			ax = plt.subplot(111)
			normVelocity = self.velocity.toarray() / norm(self.velocity.toarray())
			offset = 0.1 * normVelocity
			position = self.position.toarray() - offset
			ax.text(position[0], position[1], "T"+str(index), 
				fontsize=8, horizontalalignment = "center", verticalalignment = "center")

class Target:
	def __init__(self, initialTarget, scanIndex,measurementNumber,measurement,
				 initialState, initialStateCovariance, Parent = None, cummulativeNLLR = 0.0, residualCovariance = None):
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
			np.set_printoptions(precision = 4)
			predStateStr = " \tPredState: " + str(self.predictedStateMean)
		else:
			predStateStr = ""

		if self.measurement is not None:
			measStr = " \tMeasurement: " + str(self.measurement)
		else:
			measStr = ""

		return (repr(self.initial) 
				+ " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR)
				+ predStateStr
				+ measStr
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

	def predictMeasurement(self, Phi, Q, b, Gamma):
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(Phi,Gamma.dot(Q.dot(Gamma.T)),b,self.filteredStateMean,self.filteredStateCovariance))

	def gateAndCreateNewHypotheses(self, measurementList, sigma, P_d, lambda_ex, scanIndex, C, R, d, usedMeasurements):
		time = measurementList.time
		measurements = measurementList.measurements
		predictedPosition = Position(self.predictedStateMean[0], self.predictedStateMean[1])
		zeroInitTarget = InitialTarget(predictedPosition, self.initial.velocity, time)
		nllr = hpf.NLLR(0, P_d)
		self.trackHypotheses.append( Target(zeroInitTarget, scanIndex, 0, None, 
							zeroInitTarget.state(), self.predictedStateCovariance, self, self.cummulativeNLLR + nllr)) #Zero-hypothesis
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma, C, R):
				(_, state, filtCov, resCov) = pk._filter_correct(
					C, R, d, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				filteredPosition = Position(state[0], state[1])
				deltaPosArray = state[0:2] - self.initial.position.toarray()
				deltaTime = time - self.initial.time
				velocityArray = deltaPosArray / deltaTime
				ADHOCvelocity = Velocity(velocityArray[0], velocityArray[1])
				KFvelocity = Velocity(state[2],state[3])
				virtualInitialTarget = InitialTarget(filteredPosition,KFvelocity, time)
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
