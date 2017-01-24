# cython: profile=True

import pymht.utils.kalmanFilter as kf
import pymht.utils.helpFunctions as hpf
from pymht.utils.classDefinitions import Position, Velocity
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

class Target():
	def __init__(self, **kwargs):
		time 						= kwargs.get("time")
		scanNumber 					= kwargs.get("scanNumber")
		filteredStateMean 			= kwargs.get("filteredStateMean")
		filteredStateCovariance 	= kwargs.get("filteredStateCovariance")
		Phi							= kwargs.get("Phi")
		Q 							= kwargs.get("Q")
		Gamma 						= kwargs.get("Gamma")
		C 							= kwargs.get("C")
		R 							= kwargs.get("R")
		if ((time is None) or 
			(scanNumber is None) or 
			(filteredStateMean is None) or 
			(filteredStateCovariance is None)):
			raise TypeError("Target() need at least: time, scanNumber, state and covariance")
		#Track parameters
		self.time 	 					= time
		self.scanNumber 				= scanNumber
		self.parent 					= kwargs.get("parent")
		self.measurementNumber 			= kwargs.get("measurementNumber", 0)
		self.measurement 				= kwargs.get("measurement")
		self.cummulativeNLLR 			= kwargs.get("cummulativeNLLR", 0)
		self.trackHypotheses 			= []	

		#Kalman filter variables
		##Parent KF "measurement update"
		self.measurementResidual 		= kwargs.get("measurementResidual")
		self.residualCovariance 		= kwargs.get("residualCovariance")
		self.kalmanGain 				= kwargs.get("kalmanGain")
		self.filteredStateMean 			= filteredStateMean
		self.filteredStateCovariance 	= filteredStateCovariance
		##Current KF "time update"
		self.predictedStateMean 		= None
		self.predictedStateCovariance 	= None
		
		#State space model
		self.Phi						= Phi
		self.Q 							= Q
		self.Gamma 						= Gamma
		self.C 							= C
		self.R 							= R

	def __repr__(self):
		if self.predictedStateMean is not None:
			np.set_printoptions(precision = 4, suppress = True)
			predStateStr = " \tPredState: " + str(self.predictedStateMean)
		else:
			predStateStr = ""

		if self.measurementNumber is not None:
			measStr = (" \tMeasurement("+str(self.scanNumber)
						+":"+str(self.measurementNumber) + ")")
			if self.measurement is not None:
				measStr += ":" + str(self.measurement)
		else:
			measStr = ""

		if self.residualCovariance is not None:
			lambda_, _ = np.linalg.eig(self.residualCovariance)
			gateStr = (" \tGate size: ("+'{:5.2f}'.format(np.sqrt(lambda_[0])*2)
						+","+'{:5.2f}'.format(np.sqrt(lambda_[1])*2)+")")
		else:
			gateStr = ""

		return ("Time: " + time.strftime("%H:%M:%S", time.gmtime(self.time))
				+ "\t"  + str(self.getPosition())
				+ " \t" + str(self.getVelocity()) 
				+ " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR)
				+ measStr
				+ predStateStr
				+ gateStr 
				)

	def __str__(self, **kwargs):
		level 		= kwargs.get("level", 0)
		hypIndex 	= kwargs.get("hypIndex", 0)
		targetIndex = kwargs.get("targetIndex","?")

		if (level == 0) and not self.trackHypotheses:
			return repr(self)
		ret = ""
		if level == 0:
			ret += "T" + str(targetIndex) + ": " + repr(self) + "\n"
		else:
			ret += "   " + " "*min(level,8) + "H" + str(hypIndex)+": " +repr(self)+"\n"
		for hypIndex, hyp in enumerate(self.trackHypotheses):
			hasNotZeroHyp = (self.trackHypotheses[0].measurementNumber != 0)
			ret += hyp.__str__(level = level+1, hypIndex = hypIndex + int(hasNotZeroHyp))
		return ret

	def __sub__(self,other):
		return self.filteredStateMean - other.filteredStateMean

	def getPosition(self):
		pos = Position(self.filteredStateMean[0:2])
		return pos

	def getVelocity(self):
		return Velocity(self.filteredStateMean[2:4])

	def stepBack(self,stepsBack = 1):
		if (stepsBack == 0) or (self.parent is None):
			return self
		return self.parent.stepBack(stepsBack-1)

	def getRoot(self):
		return self.stepBack(float('inf'))

	def depth(self, count = 0):
		if len(self.trackHypotheses):
			return self.trackHypotheses[0].depth(count +1)
		return count

	def predictMeasurement(self, scanTime):
		dT = scanTime - self.time
		self.predictedStateMean, self.predictedStateCovariance = (
			kf.filterPredict(
				self.Phi(dT),
				self.Gamma.dot(self.Q(dT).dot(self.Gamma.T)),
				self.filteredStateMean,
				self.filteredStateCovariance)
			)
		self.predictedMeasurement = self.C.dot(self.predictedStateMean)
		self.residualCovariance = self.C.dot(self.predictedStateCovariance.dot(self.C.T))+self.R
	
	def gateAndCreateNewHypotheses(self, measurementList,scanNumber, P_d, lambda_ex, eta2, **kwargs):
		tic = time.process_time()
		assert self.scanNumber == scanNumber-1,"gateAndCreateNewMeasurement: inconsistent scan numbering"
		scanTime = measurementList.time
		associatedMeasurements = set()
		trackHypotheses = list()

		trackHypotheses.append( self.createZeroHypothesis(scanTime, scanNumber, P_d, **kwargs) )

		#print("measurements")
		#print(measurementList.measurements)
		#print("predicted measurement", self.predictedMeasurement)
		measurementsResidual = measurementList.measurements - self.predictedMeasurement
		#print("measurement residual\n", measurementsResidual)
		self.invResidualCovariance = np.linalg.inv(self.residualCovariance)
		#print("inverse residual covaiance\n",self.invResidualCovariance)
		
		normalizedInnovationSquared = np.zeros(len(measurementList.measurements))
		for i, residual in enumerate(measurementsResidual):
			normalizedInnovationSquared[i] = residual.T.dot(self.invResidualCovariance).dot(residual) #TODO: Vectorize this!
		#print("NIS", *normalizedInnovationSquared, sep = "\n", end = "\n\n")

		gatedMeasurements = normalizedInnovationSquared <= eta2
		#print("gated measurements", *gatedMeasurements, sep = "\n", end = "\n\n")

		for measurementIndex, insideGate in enumerate(gatedMeasurements):
			if insideGate:
				measurementResidual = measurementsResidual[measurementIndex]
				measurement = measurementList.measurements[measurementIndex]
				filtState, filtCov = kf.filterCorrect(
					self.predictedStateCovariance,self.C, self.invResidualCovariance,self.predictedStateMean, measurementResidual)
				associatedMeasurements.add( (scanNumber, measurementIndex+1) )
				trackHypotheses.append(
					self.clone(
						time 					= scanTime, 
						scanNumber 				= scanNumber,
						measurementNumber 		= measurementIndex+1,
						measurement 			= measurement,
						filteredStateMean 		= filtState,
						filteredStateCovariance	= filtCov,
						cummulativeNLLR 		= self.calculateCNLLR(P_d, measurement, lambda_ex, self.residualCovariance),
						measurementResidual 	= measurementResidual,
						)
					)
		toc = time.process_time() - tic
		return trackHypotheses, associatedMeasurements, tic
	
	def calculateCNLLR(self, P_d, measurement, lambda_ex, resCov):
		return 	(self.cummulativeNLLR +
					hpf.nllr(	P_d, 
								measurement,
								np.dot(self.C,self.predictedStateMean),
								lambda_ex,
								resCov)
				)

	def clone(self, **kwargs):
		time						=	kwargs.get("time")
		scanNumber					=	kwargs.get("scanNumber")
		measurementResidual			= 	kwargs.get("measurementResidual")
		residualCovariance 			= 	kwargs.get("residualCovariance")
		kalmanGain 					= 	kwargs.get("kalmanGain")
		filteredStateMean			=	kwargs.get("filteredStateMean")
		filteredStateCovariance		=	kwargs.get("filteredStateCovariance")
		cummulativeNLLR				=	kwargs.get("cummulativeNLLR")
		measurementNumber			=	kwargs.get("measurementNumber")
		measurement					=	kwargs.get("measurement")
		parent						=	kwargs.get("parent", self)
		Phi							=	kwargs.get("Phi",	self.Phi)
		Q							=	kwargs.get("Q",		self.Q)
		Gamma						=	kwargs.get("Gamma",	self.Gamma)
		C							=	kwargs.get("C",		self.C)
		R							=	kwargs.get("R",		self.R)

		return Target(
			time 	 					= time,
			scanNumber 					= scanNumber,
			filteredStateMean			= filteredStateMean,
			filteredStateCovariance		= filteredStateCovariance,
			parent 						= parent,
			measurementNumber 			= measurementNumber,
			measurement 				= measurement,
			cummulativeNLLR 			= cummulativeNLLR,
			Phi							= Phi,
			Q 							= Q,
			Gamma 						= Gamma,
			C 							= C,
			R 							= R,
			)

	def measurementIsInsideErrorEllipse(self,measurement, eta2):
		measRes = measurement.position - self.predictedMeasurement
		return measRes.T.dot(self.invResidualCovariance).dot(measRes) <= eta2

	def createZeroHypothesis(self,time, scanNumber, P_d, **kwargs):
		return self.clone(	time 					= time,
							scanNumber 				= scanNumber, 
							measurementNumber 		= 0,
							filteredStateMean 		= self.predictedStateMean, 
							filteredStateCovariance = self.predictedStateCovariance, 
							cummulativeNLLR 		= self.cummulativeNLLR + hpf.nllr(P_d),
							parent 					= kwargs.get("parent", self)
						)

	def createZeroHypothesisDictionary(self, time, scanNumber, P_d, **kwargs):
		return self.__dict__

	def _pruneAllHypothesisExeptThis(self, keep):
		for hyp in self.trackHypotheses:
			if hyp != keep:
				self.trackHypotheses.remove(hyp)

	def getMeasurementSet(self, root = True):
		subSet = set()
		for hyp in self.trackHypotheses:
			subSet |= hyp.getMeasurementSet(False) 
		if (self.measurementNumber == 0) or (root):
			return subSet
		else:
			return {(self.scanNumber, self.measurementNumber)} | subSet

	def processNewMeasurement(self, measurementList, measurementSet,scanNumber, P_d, lambda_ex, eta2):
		if not self.trackHypotheses:
			self.predictMeasurement(measurementList.time)
			trackHypotheses, newMeasurements,_ = self.gateAndCreateNewHypotheses(measurementList, scanNumber, P_d, lambda_ex, eta2)
			self.trackHypotheses = trackHypotheses
			measurementSet.update( newMeasurements )
		else:
			for hyp in self.trackHypotheses:
				hyp.processNewMeasurement(measurementList,measurementSet, scanNumber, P_d, lambda_ex, eta2)

	def _selectBestHypothesis(self):
		def recSearchBestHypothesis(target,bestScore, bestHypothesis):
			if len(target.trackHypotheses) == 0:
				if target.cummulativeNLLR <= bestScore[0]:
					bestScore[0] = target.cummulativeNLLR
					bestHypothesis[0] = target
			else:
				for hyp in target.trackHypotheses:
					recSearchBestHypothesis(hyp, bestScore, bestHypothesis)
		bestScore = [float('Inf')]
		bestHypothesis = np.empty(1,dtype = np.dtype(object))
		recSearchBestHypothesis(self, bestScore, bestHypothesis)
		return bestHypothesis

	def getLeafNodes(self):
		def recGetLeafNode(node, nodes):
			if not node.trackHypotheses:
				nodes.append(node)
			else:
				for hyp in node.trackHypotheses:
					recGetLeafNode(hyp,nodes)
		nodes = []
		recGetLeafNode(self,nodes)
		return nodes

	def getLeafParents(self):
		leafNodes = self.getLeafNodes()
		parents = set()
		for node in leafNodes:
			parents.add(node.parent)
		return parents

	def pruneSimilarState(self, threshold):
		for hyp in self.trackHypotheses[1:]:
			deltaPos = np.linalg.norm(self.trackHypotheses[0]-hyp)
			if deltaPos <= threshold:
				self.trackHypotheses.pop(0)
				break

	def recursiveSubtractScore(self, score):
		if score == 0:
			return
		self.cummulativeNLLR -= score
		
		for hyp in self.trackHypotheses:
			hyp.recursiveSubtractScore(score)

	def _checkScanNumberIntegrety(self):
		assert type(self.scanNumber) is int, "self.scanNumber is not an integer %r" % self.scanNumber

		if self.parent is not None:
			assert type(self.parent.scanNumber) is int, "self.parent.scanNumber is not an integer %r" % self.parent.scanNumber
			assert self.parent.scanNumber  == self.scanNumber -1, "self.parent.scanNumber(%r) == self.scanNumber-1(%r)" % (self.parent.scanNumber, self.scanNumber)

		for hyp in self.trackHypotheses:
			hyp._checkScanNumberIntegrety()

	def _checkReferenceIntegrety(self):
		def recCheckReferenceIntegrety(target):
			for hyp in target.trackHypotheses:
				assert hyp.parent == target, "Inconsistent parent <-> child reference: Measurement("+str(target.scanNumber)+":"+str(target.measurementNumber)+") <-> "+"Measurement("+str(hyp.scanNumber)+":"+str(hyp.measurementNumber)+")"
				recCheckReferenceIntegrety(hyp)
		recCheckReferenceIntegrety(self.getRoot())

	def plotValidationRegion(self, eta2, stepsBack = 0):
		if self.residualCovariance is not None:
			self._plotCovarianceEllipse(eta2)
		if (self.parent is not None) and (stepsBack > 0):
			self.parent.plotValidationRegion(eta2,stepsBack-1)

	def _plotCovarianceEllipse(self, eta2):
		from matplotlib.patches import Ellipse
		lambda_, _ = np.linalg.eig(self.residualCovariance)
		ell = Ellipse( xy	 = (self.predictedStateMean[0], self.predictedStateMean[1]), 
					   width = np.sqrt(lambda_[0])*np.sqrt(eta2)*2,
					   height= np.sqrt(lambda_[1])*np.sqrt(eta2)*2,
					   angle = np.rad2deg( np.arctan2( lambda_[1], lambda_[0]) ),
					   linewidth = 2,
					   )
		ell.set_facecolor('none')
		ell.set_linestyle("dotted")
		ell.set_alpha(0.5)
		ax = plt.subplot(111)
		ax.add_artist(ell)

	def backtrackPosition(self, stepsBack = float('inf')):
		if self.parent is None:
			return [self.getPosition()]
		else:
			return self.parent.backtrackPosition(stepsBack) + [self.getPosition()]

	def plotTrack(self, stepsBack = float('inf'), **kwargs):		
		colors = itertools.cycle(["r", "b", "g"])
		track = self.backtrackPosition(stepsBack)
		plt.plot([p.x() for p in track], [p.y() for p in track], **kwargs)

	def plotMeasurement(self, stepsBack = 0, **kwargs):
		if self.measurement is not None:
			self.measurement.plot(self.measurementNumber, self.scanNumber,**kwargs)
		elif kwargs.get("dummy",False):
			self.getPosition().plot(self.measurementNumber, self.scanNumber,**kwargs)

		if (self.parent is not None) and (stepsBack > 0):
			self.parent.plotMeasurement(stepsBack-1, **kwargs)

	def plotVelocityArrow(self, stepsBack = 1):
		if self.predictedStateMean is not None:
			ax = plt.subplot(111)
			deltaPos = self.predictedStateMean[0:2] - self.filteredStateMean[0:2]
			ax.arrow(self.filteredStateMean[0], self.filteredStateMean[1], deltaPos[0], deltaPos[1],
			head_width=0.1, head_length=0.1, fc= "None", ec='k', 
			length_includes_head = "true", linestyle = "-", alpha = 0.3, linewidth = 1)
		if (self.parent is not None) and (stepsBack > 0):
			self.parent.plotVelocityArrow(stepsBack-1)