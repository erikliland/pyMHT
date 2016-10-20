import numpy as np
import helpFunctions as hpf
import pykalman.standard as pk
import matplotlib.pyplot as plt

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

# class InitialTarget:
# 	def __init__(self, *args, **kwargs):
# 		pos = kwargs.get('Position')
# 		vel = kwargs.get('Velocity')
# 		time= kwargs.get('Time')
# 		if (pos is not None) and (vel is not None) and (time is not None):
# 			self.position 	= pos
# 			self.velocity 	= vel
# 			self.time 		= time
# 		elif len(args) == 2:
# 			self.position 	= Position(args[0][0:2])
# 			self.velocity 	= Velocity(args[0][2:4])
# 			self.time 		= args[1]
# 		elif len(args) == 3:
# 			self.position 	= args[0]
# 			self.velocity 	= args[1]
# 			self.time 		= args[2]
# 		else:
# 			print("Invalid arguments to Position")
		

# 	def __str__(self):
# 		from time import gmtime, strftime
# 		timeString = strftime("%H:%M:%S", gmtime(self.time))
# 		return ("Time: "+ timeString +" ,\t"+repr(self.position) + ",\t" + repr(self.velocity))
	
# 	__repr__ = __str__

# 	def state(self):
# 		from numpy import array
# 		return array([self.position.x, self.position.y, self.velocity.x, self.velocity.y])

class Target:
	def __init__(self, **kwargs):
		time 						= kwargs.get("time")
		scanNumber 					= kwargs.get("scanNumber")
		filteredStateMean 			= kwargs.get("state")
		filteredStateCovariance 	= kwargs.get("covariance")
		if (time is None) or (scanNumber is None) or (filteredStateMean is None) or (filteredStateCovariance is None):
			raise TypeError("Targer() need at least: time, scanNumber, state and covariance")
		self.time 						= time
		self.scanNumber 				= scanNumber
		self.filteredStateMean 			= filteredStateMean
		self.filteredStateCovariance 	= filteredStateCovariance
		self.parent 					= kwargs.get("parent")
		self.measurementNumber 			= kwargs.get("measurementNumber", 0)
		self.measurement 				= kwargs.get("measurement")
		self.cummulativeNLLR 			= kwargs.get("cummulativeNLLR", 0)
		self.predictedStateMean 		= None
		self.predictedStateCovariance 	= None
		self.residualCovariance 		= None
		self.trackHypotheses 			= []

	def __repr__(self):
		from time import gmtime, strftime
		if self.predictedStateMean is not None:
			np.set_printoptions(precision = 4)
			predStateStr = " \tPredState: " + str(self.predictedStateMean)
		else:
			predStateStr = ""

		if self.measurement is not None:
			measStr = " \tMeasurement: " + str(self.measurement)
		else:
			measStr = ""

		if self.residualCovariance is not None:
			lambda_, _ = np.linalg.eig(self.residualCovariance)
			gateStr = " \tGate size: ("+'{:5.2f}'.format(np.sqrt(lambda_[0])*2)+","+'{:5.2f}'.format(np.sqrt(lambda_[1])*2)+")"
		else:
			gateStr = ""

		return ("Time: " + strftime("%H:%M:%S", gmtime(self.time))
				+ " \t" + repr(self.getPosition())
				+ " \t" + repr(self.getVelocity()) 
				+ " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR)
				+ predStateStr
				+ measStr
				+ gateStr 
				)

	def __str__(self, level=0, hypIndex = 0):
		ret = ""
		if level != 0:
			ret += "\t" + "\t"*level + "H" + str(hypIndex)+":\t" +repr(self)+"\n"
		for hypIndex, hyp in enumerate(self.trackHypotheses):
			ret += hyp.__str__(level+1, hypIndex)
		return ret

	def getPosition(self):
		pos = Position(self.filteredStateMean[0:2])
		return pos

	def getVelocity(self):
		return Velocity(self.filteredStateMean[2:4])

	def depth(self, count = 0):
		if len(self.trackHypotheses):
			return self.trackHypotheses[0].depth(count +1)
		return count

	def predictMeasurement(self, Phi, Q, b, Gamma, C, R):
		self.predictedStateMean, self.predictedStateCovariance = (
				pk._filter_predict(Phi,Gamma.dot(Q.dot(Gamma.T)),b,self.filteredStateMean,self.filteredStateCovariance))
		self.residualCovariance = C.dot(self.predictedStateCovariance.dot(C.T))+R
	
	def gateAndCreateNewHypotheses(self, measurementList, sigma, P_d, lambda_ex, scanNumber, C, R, d, usedMeasurements):
		time = measurementList.time
		
		zeroHypothesis = self._generateZeroHypothesis(time, scanNumber, P_d)
		self.trackHypotheses.append(zeroHypothesis)
		
		for measurementIndex, measurement in enumerate(measurementList.measurements):
			if self.measurementIsInsideErrorEllipse(measurement, sigma):
				(_, filtState, filtCov, resCov) = pk._filter_correct(C, R, d, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				predictedMeasurement = np.dot(C,self.predictedStateMean)
				NLLR = hpf.nllr(P_d, measurement, predictedMeasurement, lambda_ex, resCov)
				self.trackHypotheses.append( 
					Target(	time = time, 
							scanNumber = scanNumber,
							measurementNumber = measurementIndex+1,
							measurement = measurement,
							state = filtState,
							covariance = filtCov,
							parent = self,
							cummulativeNLLR = self.cummulativeNLLR + NLLR)
							)
				usedMeasurements[measurementIndex] = True

	def measurementIsInsideErrorEllipse(self,measurement, sigma):
		from numpy import cos, sin, power
		lambda_, v = np.linalg.eig(self.residualCovariance)
		deltaX = measurement.x - self.predictedStateMean[0]
		deltaY = measurement.y - self.predictedStateMean[1]
		a = np.sqrt(lambda_[0])*sigma
		b = np.sqrt(lambda_[1])*sigma
		angle = np.rad2deg( np.arctan2( lambda_[1], lambda_[0]) )
		return power( cos(angle)*deltaX+sin(angle)*deltaY,2 )/power(a,2)  + power( sin(angle)*deltaX-cos(angle)*deltaY,2)/power(b,2) <= 1
		#http://stackoverflow.com/questions/7946187/point-and-ellipse-rotated-position-test-algorithm

	def _generateZeroHypothesis(self,time, scanNumber, P_d):
		NLLR = hpf.nllr(P_d)
		return Target(	time = time,
						scanNumber = self.scanNumber, 
						measurementNumber = 0,
						state = self.predictedStateMean, 
						covariance = self.predictedStateCovariance, 
						parent = self,
						cummulativeNLLR = self.cummulativeNLLR + NLLR
						)

	def plotInitial(self, index):
		plt.plot(self.filteredStateMean[0],self.filteredStateMean[1],"k+")
		ax = plt.subplot(111)
		normVelocity = self.filteredStateMean[2:4] / np.linalg.norm(self.filteredStateMean[2:4])
		offset = 0.1 * normVelocity
		position = self.filteredStateMean[0:2] - offset
		ax.text(position[0], position[1], "T"+str(index), 
			fontsize=8, horizontalalignment = "center", verticalalignment = "center")