#!/usr/bin/env python3
"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Authumn 2016
========================================================================================
"""

from . import helpFunctions as hpf
from . import kalmanFilter as kf
from .classDefinitions import Position, Velocity
import time
import pulp
import itertools
import logging
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

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
			ret += "T" + str(targetIndex) + ": \t" + repr(self) + "\n"
		else:
			ret += "\t" + "\t"*min(level,5) + "H" + str(hypIndex)+":\t" +repr(self)+"\n"
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
		self.residualCovariance = self.C.dot(
						self.predictedStateCovariance.dot(self.C.T))+self.R
		self.observableState = self.C.dot(self.predictedStateMean)
	
	def gateAndCreateNewHypotheses(self, measurementList, associatedMeasurements, tracker):
		scanNumber = len(tracker.__scanHistory__)
		assert self.scanNumber == scanNumber-1,"gateAndCreateNewMeasurement: inconsistent scan numbering"
		P_d = tracker.P_d
		lambda_ex = tracker.lambda_ex
		time = measurementList.time
		self.addZeroHypothesis(time, scanNumber, P_d)

		for measurementIndex, measurement in enumerate(measurementList.measurements):
			if self.measurementIsInsideErrorEllipse(measurement,tracker.eta2):
				(measRes, resCov, kalmanGain, filtState, filtCov) = kf.filterCorrect(
					self.C, self.R, self.predictedStateMean, self.predictedStateCovariance, measurement.toarray() )
				associatedMeasurements.add( (scanNumber, measurementIndex+1) )
				self.trackHypotheses.append(
					self.clone(
						time 					= time, 
						scanNumber 				= scanNumber,
						measurementNumber 		= measurementIndex+1,
						measurement 			= measurement,
						filteredStateMean 		= filtState,
						filteredStateCovariance	= filtCov,
						cummulativeNLLR 		= self.calculateCNLLR(P_d, measurement, lambda_ex, resCov),
						measurementResidual 	= measRes,
						residualCovariance 		= resCov,
						kalmanGain 				= kalmanGain
						)
					)
	
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
		parent						=	kwargs.get("parent",self)
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
		measRes = measurement.toarray()- self.observableState
		#measRes = measurement.toarray()- self.C.dot(self.predictedStateMean)
		return measRes.T.dot( np.linalg.inv(self.residualCovariance).dot( measRes ) ) <= eta2

	def addZeroHypothesis(self,time, scanNumber, P_d):
		self.trackHypotheses.append(
			self.clone(	time 					= time,
						scanNumber 				= scanNumber, 
						measurementNumber 		= 0,
						filteredStateMean 		= self.predictedStateMean, 
						filteredStateCovariance = self.predictedStateCovariance, 
						cummulativeNLLR 		= self.cummulativeNLLR + hpf.nllr(P_d)
						)
			)

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

	def processNewMeasurement(self, measurementList, measurementSet,tracker):
		if not self.trackHypotheses:
			self.predictMeasurement(measurementList.time)
			self.gateAndCreateNewHypotheses(measurementList,measurementSet, tracker)
		else:
			for hyp in self.trackHypotheses:
				hyp.processNewMeasurement(measurementList,measurementSet, tracker)

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
			if len(node.trackHypotheses) == 0:
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
		plt.plot([p.x for p in track], [p.y for p in track], **kwargs)

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

class Tracker():
	def __init__(self, Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, 
		lambda_nu, eta2, N, solverStr, **kwargs):

		self.logTime 	= kwargs.get("logTime", False)
		self.debug 		= kwargs.get("debug", False)

		#Tracker storage
		self.__targetList__ 			= []
		self.__scanHistory__ 			= []
		self.__associatedMeasurements__ = []
		self.__trackNodes__ 			= np.empty(0,dtype = np.dtype(object))
		self.runtimeLog = {	'Process':	np.array([0.0,0]),
							'Cluster':	np.array([0.0,0]),
							'Optim':	np.array([0.0,0]),
							'Prune':	np.array([0.0,0]),
							}
		
		#Tracker parameters
		self.P_d 		= P_d
		self.lambda_phi = lambda_phi		
		self.lambda_nu 	= lambda_nu		
		self.lambda_ex 	= lambda_phi+lambda_nu
		self.eta2		= eta2
		self.N 		 	= N
		self.solver  	= hpf.parseSolver(solverStr)
		self.pruneThreshold = kwargs.get("pruneThreshold")

		#State space model
		self.Phi 		= Phi
		self.b 			= np.zeros(4) 			
		self.C 			= C
		self.d 			= np.zeros(2)			
		self.Gamma 		= Gamma
		self.P0 		= P0
		self.R 			= R	
		self.Q			= Q

		if (kwargs.get("realTime") is not None) and (kwargs.get("realTime") is True):
			self._setHightPriority()

	def initiateTarget(self,newTarget):
		target = Target(	time 					= newTarget.time, 
							scanNumber 				= len(self.__scanHistory__),
							filteredStateMean 		= newTarget.state, 
							filteredStateCovariance = self.P0,
							Phi						= self.Phi,
							Q  						= self.Q,
							Gamma 					= self.Gamma,
							C 						= self.C,
							R 						= self.R
							)
		self.__targetList__.append(target)
		self.__associatedMeasurements__.append( set() )
		self.__trackNodes__ = np.append(self.__trackNodes__,target)

	def addMeasurementList(self,measurementList, **kwargs):
		tic1 = time.process_time()
		
		self._checkTrackerIntegrety()

		tic2 = time.process_time()
		self.__scanHistory__.append(measurementList)
		nMeas = len(measurementList.measurements)
		nTargets = len(self.__targetList__)
		for targetIndex, target in enumerate(self.__targetList__):
			target.processNewMeasurement(measurementList, self.__associatedMeasurements__[targetIndex],self)
		toc2 = time.process_time() - tic2
		if kwargs.get("printAssociation",False):
			print(*__associatedMeasurements__, sep = "\n", end = "\n\n")

		#--Cluster targets--
		tic3 = time.process_time()
		clusterList = self._findClustersFromSets()
		toc3 = time.process_time() - tic3
		if kwargs.get("printCluster",False):
			hpf.printClusterList(clusterList)
		
		#--Maximize global (cluster vise) likelihood--
		tic4 = time.time()
		nOptimSolved = 0
		for cluster in clusterList:
			if len(cluster) == 1:
				#self._pruneSmilarState(cluster, self.pruneThreshold)
				self.__trackNodes__[cluster] = self.__targetList__[cluster[0]]._selectBestHypothesis()
			else:
				#self._pruneSmilarState(cluster, self.pruneThreshold/2)
				self.__trackNodes__[cluster] = self._solveOptimumAssociation(cluster)
				nOptimSolved += 1
		toc4 = time.time()-tic4

		tic5 = time.process_time()
		self._nScanPruning()
		toc5 = time.process_time()-tic5
		toc1 = time.process_time() - tic1

		self._checkTrackerIntegrety()

		if self.logTime:
			self.runtimeLog['Process'] 	+= np.array([toc2,1])
			self.runtimeLog['Cluster'] 	+= np.array([toc3,1])
			self.runtimeLog['Optim'] 	+= np.array([toc4,1])
			self.runtimeLog['Prune']	+= np.array([toc5,1])

		if kwargs.get("printTime",False):
			print(	"Added scan number:", len(self.__scanHistory__),
					" \tnMeas ", nMeas,
					" \tTotal time ", '{:5.4f}'.format(toc1),
					"\tProcess ",	'{:5.4f}'.format(toc2),
					"\tCluster ",	'{:5.4f}'.format(toc3),
					'\tOptim({0:g}) {1:5.4f}'.format(nOptimSolved ,toc4),
					"\tPrune ",	'{:5.4f}'.format(toc5),
					sep = "")

		#Covariance consistance
		if "trueState" in kwargs:
			xTrue = kwargs.get("trueState")
			return [(target.filteredStateMean-xTrue[targetIndex].state).T.dot(
					np.linalg.inv(target.filteredStateCovariance)).dot(
					(target.filteredStateMean-xTrue[targetIndex].state) )
			  for targetIndex, target in enumerate(self.__trackNodes__)]

	def getRuntimeAverage(self, **kwargs):
		p = kwargs.get("precision", 3)
		return {k: v[0]/v[1] for k,v in self.runtimeLog.items()}

	def _findClustersFromSets(self):
		superSet = set()
		for targetSet in self.__associatedMeasurements__:
			superSet |= targetSet
		nTargets = len(self.__associatedMeasurements__)
		nNodes = nTargets + len(superSet)
		adjacencyMatrix  = np.zeros((nNodes,nNodes),dtype=bool)
		for targetIndex, targetSet in enumerate(self.__associatedMeasurements__):
			for measurementIndex, measurement in enumerate(superSet):
				adjacencyMatrix[targetIndex,measurementIndex+nTargets] = (measurement in targetSet)
		(nClusters, labels) = connected_components(adjacencyMatrix)
		return [np.where(labels[:nTargets]==clusterIndex)[0] for clusterIndex in range(nClusters)]

	def getTrackNodes(self):
		return self.__trackNodes__

	def _solveOptimumAssociation(self, cluster):
		nHypInClusterArray = self._getHypInCluster(cluster)
		nRealMeasurementsInCluster= len(set.union(*[self.__associatedMeasurements__[i] for i in cluster]))
		problemSize = nRealMeasurementsInCluster*sum(nHypInClusterArray)

		t0 = time.process_time()
		(A1, measurementList) = self._createA1(nRealMeasurementsInCluster,sum(nHypInClusterArray), cluster)
		A2 	= self._createA2(len(cluster), nHypInClusterArray)
		C 	= self._createC(cluster)
		t1 = time.process_time()-t0
		# print("matricesTime\t", round(t1,3))

		selectedHypotheses = self._solveBLP(A1,A2, C, len(cluster))
		selectedNodes = self._hypotheses2Nodes(selectedHypotheses,cluster)
		selectedNodesArray = np.array(selectedNodes)
		# print("Solving optimal association in cluster with targets",cluster,",   \t",
		# sum(nHypInClusterArray)," hypotheses and",nRealMeasurementsInCluster,"real measurements.",sep = " ")
		# print("nHypothesesInCluster",sum(nHypInClusterArray))
		# print("nRealMeasurementsInCluster", nRealMeasurementsInCluster)	
		# print("nTargetsInCluster", len(cluster))
		# print("nHypInClusterArray",nHypInClusterArray)
		# print("c =", c)
		# print("A1", A1, sep = "\n")
		# print("size(A1)", A1.shape, "\t=>\t", nRealMeasurementsInCluster*sum(nHypInClusterArray))
		# print("A2", A2, sep = "\n")
		# print("measurementList",measurementList)
		# print("selectedHypotheses",selectedHypotheses)
		# print("selectedNodes",*selectedNodes, sep = "\n")
		# print("selectedNodesArray",*selectedNodesArray, sep = "\n")

		assert len(selectedHypotheses) == len(cluster), "__solveOptimumAssociation did not find the correct number of hypotheses"
		assert len(selectedNodes) == len(cluster), "__solveOptimumAssociation did not find the correct number of nodes"
		assert len(selectedHypotheses) == len(set(selectedHypotheses)), "_solveOptimumAssociation selected two or more equal hyptheses"
		assert len(selectedNodes) == len(set(selectedNodes)), "_solveOptimumAssociation found same node in more than one track in selectedNodes"
		assert len(selectedNodesArray) == len(set(selectedNodesArray)), "_solveOptimumAssociation found same node in more than one track in selectedNodesArray"
		return selectedNodesArray

	def _getHypInCluster(self,cluster):
		def nLeafNodes(target):
			if not target.trackHypotheses:
				return 1
			else:
				return sum(nLeafNodes(hyp) for hyp in target.trackHypotheses)
		nHypInClusterArray = np.zeros(len(cluster), dtype = int)
		for i, targetIndex in enumerate(cluster):
			nHypInTarget = nLeafNodes(self.__targetList__[targetIndex])
			nHypInClusterArray[i] = nHypInTarget
		return nHypInClusterArray

	def _createA1(self,nRow,nCol,cluster):
		def recActiveMeasurement(target, A1, measurementList,  activeMeasurements, hypothesisIndex):
			if len(target.trackHypotheses) == 0:
				if (target.measurementNumber != 0) and (target.measurementNumber is not None): #we are at a real measurement
					measurement = (target.scanNumber,target.measurementNumber)
					try:
						measurementIndex = measurementList.index(measurement)
					except ValueError:
						measurementList.append(measurement)
						measurementIndex = len(measurementList) -1
					activeMeasurements[measurementIndex] = True
					# print("Measurement list", measurementList)
					# print("Measurement index", measurementIndex)
					# print("HypInd = ", hypothesisIndex[0])
					# print("Active measurement", activeMeasurements)
				A1[activeMeasurements,hypothesisIndex[0]] = True
				hypothesisIndex[0] += 1
				
			else:
				for hyp in target.trackHypotheses:
					activeMeasurementsCpy = activeMeasurements.copy()
					if (hyp.measurementNumber != 0) and (hyp.measurementNumber is not None): 
						measurement = (hyp.scanNumber,hyp.measurementNumber)
						try:
							measurementIndex = measurementList.index(measurement)
						except ValueError:
							measurementList.append(measurement)
							measurementIndex = len(measurementList) -1
						activeMeasurementsCpy[measurementIndex] = True
					recActiveMeasurement(hyp, A1, measurementList, activeMeasurementsCpy, hypothesisIndex)
		A1 	= np.zeros((nRow,nCol), dtype = bool) #Numpy Array
		# A1 = sp.dok_matrix((nRow,nCol),dtype = bool) #pulp.sparse Matrix
		activeMeasurements = np.zeros(nRow, dtype = bool)
		measurementList = []
		hypothesisIndex = [0]
		#TODO: http://stackoverflow.com/questions/15148496/python-passing-an-integer-by-reference
		for targetIndex in cluster:
			recActiveMeasurement(self.__targetList__[targetIndex],A1,measurementList,activeMeasurements,hypothesisIndex)
		return A1, measurementList

	def _createA2(self, nTargetsInCluster, nHypInClusterArray):
		A2 	= np.zeros((nTargetsInCluster,sum(nHypInClusterArray)), dtype = bool)
		colOffset = 0
		for rowIndex, nHyp in enumerate(nHypInClusterArray):
			for colIndex in range(colOffset, colOffset + nHyp):
				A2[rowIndex,colIndex]=True
			colOffset += nHyp
		return A2

	def _createC(self,cluster):
		def getTargetScore(target, scoreArray):
			if len(target.trackHypotheses) == 0:
				scoreArray.append(target.cummulativeNLLR)
			else:
				for hyp in target.trackHypotheses:
					getTargetScore(hyp, scoreArray)
		scoreArray = []
		for targetIndex in cluster:
			getTargetScore(self.__targetList__[targetIndex], scoreArray)
		return scoreArray

	def _hypotheses2Nodes(self,selectedHypotheses, cluster):
		def recDFS(target, selectedHypothesis, nodeList, counter):
			if len(target.trackHypotheses) == 0:
				if counter[0] in selectedHypotheses:
					nodeList.append(target)
				counter[0] += 1
			else:
				for hyp in target.trackHypotheses:
					recDFS(hyp, selectedHypotheses, nodeList, counter)
		nodeList = []
		counter = [0]
		for targetIndex in cluster:
			recDFS(self.__targetList__[targetIndex], selectedHypotheses, nodeList, counter)
		return nodeList

	def _solveBLP(self, A1, A2, f, nHyp):
		(nMeas, nHyp) = A1.shape
		(nTargets, _) = A2.shape
		prob = pulp.LpProblem("Association problem", pulp.LpMinimize)
		x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
		c = pulp.LpVariable.dicts("c", range(nHyp))
		for i in range(len(f)):
			c[i] = f[i]
		prob += pulp.lpSum(c[i]*x[i] for i in range(nHyp))
		for row in range(nMeas):
			prob += pulp.lpSum([ A1[row,col] * x[col] for col in range(nHyp) if A1[row,col]]) <= 1
		for row in range(nTargets):
			prob += pulp.lpSum([ A2[row,col] * x[col] for col in range(nHyp)  if A2[row,col] ]) == 1
		tic = time.process_time()
		sol = prob.solve(self.solver)
		toc = time.process_time()-tic

		def getSelectedHyp1(p, threshold = 0):
			hyp = [int(v.name[2:]) for v in p.variables() if abs(v.varValue-1) <= threshold]
			hyp.sort()
			return hyp
		def getSelectedHyp2(p, threshold = 0):
			hyp = [int(v[0][2:]) for v in p.variablesDict().items() if abs(v[1].varValue-1) <= threshold]
			hyp.sort()
			return hyp

		for threshold in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
			selectedHypotheses = getSelectedHyp2(prob, threshold)
			if len(selectedHypotheses) == nHyp:
				break
		return selectedHypotheses

	def _nScanPruning(self):
		def recPruneNScan(node, targetIndex, targetList, stepsLeft):
			if stepsLeft <= 0:
				if node.parent is not None:
					if targetList[targetIndex].scanNumber != node.scanNumber-1:
						raise ValueError("nScanPruning1: from scanNumber",targetList[targetIndex].scanNumber,"->",node.scanNumber)
					changed = (targetList[targetIndex] != node)
					targetList[targetIndex] = node
					node.parent._pruneAllHypothesisExeptThis(node)
					node.recursiveSubtractScore(node.cummulativeNLLR)
					if node.parent.scanNumber != node.scanNumber-1:
						raise ValueError("nScanPruning2: from scanNumber",node.parent.scanNumber,"->",node.scanNumber)
					return changed
				else:
					return False
			elif node.parent is not None:
				return recPruneNScan(node.parent, targetIndex, targetList, stepsLeft-1)
			else:
				return False
		
		for targetIndex, node in enumerate(self.__trackNodes__):
			self._checkTrackerIntegrety()
			changed = recPruneNScan(node, targetIndex, self.__targetList__, self.N)
			self._checkTrackerIntegrety()
			if changed:
				self.__associatedMeasurements__[targetIndex] = self.__targetList__[targetIndex].getMeasurementSet()

	def _pruneSmilarState(self,cluster, threshold):
		for targetIndex in cluster:
			leafParents = self.__targetList__[targetIndex].getLeafParents()
			for node in leafParents:
				node.pruneSimilarState(threshold)

		# nHyp = len(target.trackHypotheses)
		# nDelta = int(hpf.binomial(nHyp,2))
		# deltaX = np.zeros([4,nDelta])
		# hypotheses = target.trackHypotheses[1:]
		# done = set()
		# for a in target.trackHypotheses[:-1]:
		# 	for b in hypotheses:
		# 		if a != b:
		# 			targetID = (a.measurementNumber,b.measurementNumber)
		# 			if targetID not in done:
		# 				deltaX[:,len(done)] = (a.filteredStateMean - b.filteredStateMean)
		# 				done.add( targetID )
		# 	hypotheses.pop(0)
		# for col in range(nDelta):
		# 	errorNorm = np.linalg.norm(deltaX[:,col])
		# 	# print("Norm",round(errorNorm,3))
		# 	if errorNorm < errorNormLimit:
		# 		# print("Found similar hypotheses")
		# 		if col < nHyp:
		# 			# print("Removing zero hypothesis")
		# 			target.trackHypotheses.pop(0)

	def _checkTrackerIntegrety(self):
		assert len(self.__trackNodes__) == len(self.__targetList__), "There are not the same number trackNodes as targets"
		assert len(self.__targetList__) == len(set(self.__targetList__)), "There are copies of targets in the target list"
		assert len(self.__trackNodes__) == len(set(self.__trackNodes__)), "There are copies of track nodes in __trackNodes__"
		for target in self.__targetList__:
			target._checkScanNumberIntegrety()

	
	def plotValidationRegionFromRoot(self, stepsBack = 1):
		def recPlotValidationRegionFromTarget(target, eta2, stepsBack):
			if not target.trackHypotheses:
				target.plotValidationRegion(eta2, stepsBack)
			else:
				for hyp in target.trackHypotheses:
					recPlotValidationRegionFromTarget(hyp, eta2, stepsBack)

		for target in self.__targetList__:
			recPlotValidationRegionFromTarget(target, self.eta2, stepsBack)

	def plotValidationRegionFromTracks(self,stepsBack = 1):
		for node in self.__trackNodes__:
			node.plotValidationRegion(self.eta2, stepsBack)

	def plotHypothesesTrack(self):
		def recPlotHypothesesTrack(target, track = [], **kwargs):
			newTrack = track[:] + [target.getPosition()]
			if not target.trackHypotheses:
				plt.plot([p.x for p in newTrack], [p.y for p in newTrack], "--", **kwargs)
			else:
				for hyp in target.trackHypotheses:
					recPlotHypothesesTrack(hyp,  newTrack, **kwargs)
		colors = itertools.cycle(["r", "b", "g"])
		for target in self.__targetList__:
			recPlotHypothesesTrack(target, c = next(colors))

	def plotActiveTracks(self):
		colors = itertools.cycle(["r", "b", "g"])
		for track in self.__trackNodes__:
			track.plotTrack(c = next(colors))

	def plotMeasurementsFromTracks(self, stepsBack = float('inf'), **kwargs):
		for node in self.__trackNodes__:
			node.plotMeasurement(stepsBack, **kwargs)

	def plotMeasurementsFromRoot(self,**kwargs):
		def recPlotMeasurements(target, plottedMeasurements, plotReal, plotDummy):
			if target.parent is not None:
				if target.measurementNumber == 0:
					if plotDummy:
						target.plotMeasurement(**kwargs)
				else:
					if plotReal:
						measurementID = (target.scanNumber,target.measurementNumber)
						if measurementID not in plottedMeasurements:
							target.plotMeasurement(**kwargs)
							plottedMeasurements.add( measurementID )
			for hyp in target.trackHypotheses:
				recPlotMeasurements(hyp, plottedMeasurements, plotReal, plotDummy)
		
		if not (("real" in kwargs) or ("dummy" in kwargs)):
			return
		plottedMeasurements = set()
		for target in self.__targetList__:
			recPlotMeasurements(target,plottedMeasurements,kwargs.get("real",True), kwargs.get("dummy",True))

	def plotScan(self, index = -1, **kwargs):
		self.__scanHistory__[index].plot(**kwargs)

	def plotVelocityArrowForTrack(self, stepsBack = 1):
		for track in self.__trackNodes__:
			track.plotVelocityArrow(stepsBack)

	def plotInitialTargets(self, **kwargs):
		initialTargets = [target.getRoot() for target in self.__targetList__]
		for i, initialTarget in enumerate(initialTargets):
			index = kwargs.get("index",list(range(len(initialTargets))))
			if len(index) != len(initialTargets):
				raise ValueError("plotInitialTargets: Need equal number of targets and indecies")
			plt.plot(initialTarget.filteredStateMean[0],initialTarget.filteredStateMean[1],"k+")
			ax = plt.subplot(111)
			normVelocity = initialTarget.filteredStateMean[2:4] / np.linalg.norm(initialTarget.filteredStateMean[2:4])
			offset = 0.1 * normVelocity
			position = initialTarget.filteredStateMean[0:2] - offset
			ax.text(position[0], position[1], "T"+str(index[i]), 
				fontsize=8, horizontalalignment = "center", verticalalignment = "center")


	def printTargetList(self, **kwargs):
		print("TargetList:")
		for targetIndex, target in enumerate(self.__targetList__):
			if kwargs.get("backtrack", False):
				print(target.stepBack().__str__(targetIndex = targetIndex)) 
			else:
				print(target.__str__(targetIndex = targetIndex)) 
		print()

	def _setHightPriority(self):
		import psutil, os, platform
		p = psutil.Process(os.getpid())
		OS = platform.system()
		if (OS == "Darwin") or (OS == "Linux"):
			p.nice(5)
			print("Nice:", p.nice())
		elif OS == "Windows":
			p.nice(psutil.HIGH_PRIORITY_CLASS)
			print("Nice:", p.nice())