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
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process
from scipy.sparse.csgraph import connected_components

class Target():
	def __init__(self, **kwargs):
		Process.__init__(self)

		time 						= kwargs.get("time")
		scanNumber 					= kwargs.get("scanNumber")
		filteredStateMean 			= kwargs.get("filteredStateMean")
		filteredStateCovariance 	= kwargs.get("filteredStateCovariance")
		Phi							= kwargs.get("Phi")	
		Q 							= kwargs.get("Q")
		Gamma 						= kwargs.get("Gamma")	
		C 							= kwargs.get("C")
		R 							= kwargs.get("R")

		if (time is None) or (scanNumber is None) or (filteredStateMean is None) or (filteredStateCovariance is None):
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
			measStr = " \tMeasurement(" + str(self.scanNumber) + ":" + str(self.measurementNumber) + ")"
			if self.measurement is not None:
				measStr += ":" + str(self.measurement)
		else:
			measStr = ""

		if self.residualCovariance is not None:
			lambda_, _ = np.linalg.eig(self.residualCovariance)
			gateStr = " \tGate size: ("+'{:5.2f}'.format(np.sqrt(lambda_[0])*2)+","+'{:5.2f}'.format(np.sqrt(lambda_[1])*2)+")"
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

	def getPosition(self):
		pos = Position(self.filteredStateMean[0:2])
		return pos

	def getVelocity(self):
		return Velocity(self.filteredStateMean[2:4])

	def backtrack(self,stepsBack):
		if stepsBack == 0:
			return self
		return self.parent.backtrack(stepsBack-1)

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
	
	def gateAndCreateNewHypotheses(self, measurementList, associatedMeasurements, tracker):
		scanNumber = len(tracker.__scanHistory__)
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
		measRes = measurement.toarray()-self.C.dot(self.predictedStateMean)
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

class Tracker():
	def __init__(self, Phi, C, Gamma, P_d, P0, R, Q, 
						lambda_phi, lambda_nu, eta2, N, solverStr, **kwargs):

		self.logTime 	= kwargs.get("logTime", False)
		self.debug 		= kwargs.get("debug", False)

		#Tracker storage
		self.__targetList__ 			= []
		self.__scanHistory__ 			= []
		self.__associatedMeasurements__ = []
		self.__trackNodes__ 			= None
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

		#State space model
		self.Phi 		= Phi
		self.b 			= np.zeros(4) 			
		self.C 			= C
		self.d 			= np.zeros(2)			
		self.Gamma 		= Gamma
		self.P0 		= P0
		self.R 			= R	
		self.Q			= Q

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

	def addMeasurementList(self,measurementList, **kwargs):
		tic1 = time.clock()
		tic2 = time.clock()
		self.__scanHistory__.append(measurementList)
		nMeas = len(measurementList.measurements)
		nTargets = len(self.__targetList__)
		# if kwargs.get("multiThread", True):
		# 	processes = []
		# 	for targetIndex, target in enumerate(self.__targetList__):
		# 		processes.append(addProcess(target, measurementList, self.__associatedMeasurements__[targetIndex],self))
		# 	for process in processes:
		# 		process.start()
		# 	for process in processes:
		# 		process.join()
		# else:
		for targetIndex, target in enumerate(self.__targetList__):
			target.processNewMeasurement(measurementList, self.__associatedMeasurements__[targetIndex],self)

		toc2 = time.clock() - tic2
		if kwargs.get("printAssociation",False):
			print(*__associatedMeasurements__, sep = "\n", end = "\n\n")
		
		#--Cluster targets--
		tic3 = time.clock()
		clusterList = self._findClustersFromSets()
		toc3 = time.clock() - tic3
		if kwargs.get("printCluster",False):
			hpf.printClusterList(clusterList)

		#--Maximize global (cluster vise) likelihood--
		tic4 = time.clock()
		self.__trackNodes__ = np.empty(len(self.__targetList__),dtype = np.dtype(object))
		for cluster in clusterList:
			if len(cluster) == 1:
				# self._pruneSmilarState(self.__targetList__[cluster[0]], 1)
				self.__trackNodes__[cluster] = self.__targetList__[cluster[0]]._selectBestHypothesis()
			else:
				self.__trackNodes__[cluster] = self._solveOptimumAssociation(cluster)
		toc4 = time.clock()-tic4
		
		tic5 = time.clock()
		self._nScanPruning()
		toc5 = time.clock()-tic5
		toc1 = time.clock() - tic1

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
					"\tOptim ",	'{:5.4f}'.format(toc4),
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
		return {k:'{1:.{0}e}'.format(p, v[0]/v[1]) for k,v in self.runtimeLog.items()}

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
		return [np.where(labels[:nTargets]==clusterIndex)[0].tolist() for clusterIndex in range(nClusters)]

	def getTrackNodes(self):
		return self.__trackNodes__

	def _solveOptimumAssociation(self, cluster):
		nHypInClusterArray = self._getHypInCluster(cluster)
		nRealMeasurementsInCluster= len(set.union(*[self.__associatedMeasurements__[i] for i in cluster]))
		problemSize = nRealMeasurementsInCluster*sum(nHypInClusterArray)
		# print("problemSize", problemSize)
		# if problemSize > problemSizeLimit:
		# 	_nScanPruning(, N-1)
		# 	nHypInClusterArray = _getHypInCluster(cluster)
		# 	nRealMeasurementsInCluster= len(set.union(*[__associatedMeasurements__[i] for i in cluster]))
		# 	print("reduced problemSize:", nRealMeasurementsInCluster*sum(nHypInClusterArray))
		# t0 = time.clock()
		(A1, measurementList) = self._createA1(nRealMeasurementsInCluster,sum(nHypInClusterArray), cluster)
		A2 	= self._createA2(len(cluster), nHypInClusterArray)
		C 	= self._createC(cluster)
		# t1 = time.clock()-t0
		# print("matricesTime\t", round(t1,3))

		selectedHypotheses = self._solveBLP(A1,A2, C)
		selectedNodes = self._hypotheses2Nodes(selectedHypotheses,cluster)
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
		# print("selectedMeasurements",selectedMeasurements)
		#return np.array(selectedMeasurements, dtype = int, ndmin = 2).T
		return np.array(selectedNodes)

	def _getHypInCluster(self,cluster):
		def nLeafNodes(target):
			if len(target.trackHypotheses) == 0:
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
		A1 	= np.zeros((nRow,nCol), dtype = bool)
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

	def _solveBLP(self, A1, A2, f):
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
		tic = time.clock()
		sol = prob.solve(self.solver)
		toc = time.clock()-tic
		if self.debug:
			print("n=",nHyp,"=>",round(toc,3))
		def getSelectedHyp1(p):
			hyp = [int(v.name[2:]) for v in p.variables() if v.varValue ==1]
			hyp.sort()
			return hyp
		def getSelectedHyp2(p):
			hyp = [int(v[0][2:]) for v in p.variablesDict().items() if v[1].varValue==1]
			hyp.sort()
			return hyp
		return getSelectedHyp2(prob)

	def _nScanPruning(self):
		#TODO: Should the cNLLR be "normalized" so that its zero at each root node?
		def recPruneNScan(target, targetIndex, targetList, stepsLeft):
			if stepsLeft <= 0:
				if target.parent is not None:
					changed = (targetList[targetIndex] != target)
					targetList[targetIndex] = target
					target.parent._pruneAllHypothesisExeptThis(target)
					return (changed, target.scanNumber)
				else:
					return (False, None)
			elif target.parent is not None:
				return recPruneNScan(target.parent, targetIndex, targetList, stepsLeft-1)
			else:
				return (False, None)
		for targetIndex, target in enumerate(self.__trackNodes__):
			(changed, scanNumber) = recPruneNScan(target, targetIndex, self.__targetList__, self.N)
			if changed:
				self.__associatedMeasurements__[targetIndex] = self.__targetList__[targetIndex].getMeasurementSet()

	def _pruneSmilarState(self,target, errorNormLimit):
		# print("Pruning")
		
		nHyp = len(target.trackHypotheses)
		nDelta = int(hpf.binomial(nHyp,2))
		deltaX = np.zeros([4,nDelta])
		hypotheses = target.trackHypotheses[1:]
		done = set()
		for a in target.trackHypotheses[:-1]:
			for b in hypotheses:
				if a != b:
					targetID = (a.measurementNumber,b.measurementNumber)
					if targetID not in done:
						deltaX[:,len(done)] = (a.filteredStateMean - b.filteredStateMean)
						done.add( targetID )
			hypotheses.pop(0)
		for col in range(nDelta):
			errorNorm = np.linalg.norm(deltaX[:,col])
			# print("Norm",round(errorNorm,3))
			if errorNorm < errorNormLimit:
				# print("Found similar hypotheses")
				if col < nHyp:
					# print("Removing zero hypothesis")
					target.trackHypotheses.pop(0)


