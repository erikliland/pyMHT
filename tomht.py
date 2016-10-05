"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Authumn 2016
========================================================================================
"""
##Initiation starting
import pykalman.standard as pk
import numpy as np
from pykalman.utils import Bunch
import helpFunctions as hpf
import pulp
import math

def Phi(T):
	from numpy import array
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])

#State space model
timeStep = 1.0 #second
T 		= timeStep
A 		= Phi(T) 					#Transition matrix
b 		= np.zeros(4) 				#Transition offsets
C 		= np.array([[1, 0, 0, 0],	#Also known as "H"
					[0, 1, 0, 0]])	
d 		= np.zeros(2)				#Observation offsets
Gamma 	= np.diag([1,1],-2)[:,0:2]	#Disturbance matrix (only velocity)
P_d 	= 0.8						#Probability of detection
p 		= math.pow(2e-2,2)			#Initial systen state variance
P0 		= np.diag([p,p,p/2,p/2])	#Initial state covariance
r		= math.pow(4e-2,1)			#Measurement variance
q 		= math.pow(4e-2,1)			#Velocity variance variance
R 		= np.eye(2) * r 			#Measurement/observation covariance
Q		= np.eye(2) * q * T 		#Transition/system covariance (process noise)
lambda_ex = 1 						#Spatial density of the extraneous measurements

__targetList__ = []
__clusterList__ = []
__lastMeasurementTime__ = -1.0
__sigma__ = 3
__scanHistory = []
##Initiation finished


def NLLR(hypothesisIndex, measurement,predictedMeasurement,lambda_ex,covariance,P_d):
	from numpy import dot, transpose, log, pi, power
	from numpy.linalg import inv, det
	measurementResidual = measurement.toarray() - predictedMeasurement
	if hypothesisIndex == 0:
		return -log(1-P_d)
	else:
		return (	0.5*(measurementResidual.T.dot(inv(covariance)).dot(measurementResidual))
					+ log((lambda_ex*power(det(2*pi*covariance),0.5))/P_d) 	)

# def getKalmanFilterInitData(initialTarget):
# 	import numpy as np
# 	return Bunch(	transitionMatrix 		= A,
# 					observationMatrix 		= C,
# 					transitionCovariance 	= Q,
# 					bbservationCovariance 	= R,
# 					transitionOffsets 		= b,
# 					observationOffsets 		= d,
# 					initialStateMean 		= initialTarget.state(),
# 					initialStateCovariance 	= P0,
# 					randomState 			= 0)

def initiateTrack(initialTarget):
	from classDefinitions import Target
	currentScanIndex = len(__scanHistory)
	tempTarget = Target(initialTarget, currentScanIndex, None, None, initialTarget.state(), P0)
	__targetList__.append(tempTarget)
	hpf.plotInitialTargetIndex(initialTarget, len(__targetList__)-1)

def addMeasurementList(measurementList):
	import numpy as np
	__scanHistory.append(measurementList)
	scanIndex = len(__scanHistory)
	print("#"*150)
	print("Adding scan index:", scanIndex)
	hpf.plotMeasurements(measurementList)
	hpf.plotMeasurementIndecies(scanIndex, measurementList.measurements)
	nMeas = len(measurementList.measurements)
	nTargets = len(__targetList__)
	associationMatrix = np.zeros((nMeas, nTargets),dtype = np.bool)
	for targetIndex, target in enumerate(__targetList__):
		processTarget(target, measurementList, scanIndex, associationMatrix[:,targetIndex])
		# hpf.printMeasurementAssociation(targetIndex, target)
	
	# print("AssosiationMatrix")
	# print(associationMatrix)
	# print()

	#cluster / merge-split clusters
	clusterList = findClusters(associationMatrix)
	# hpf.printClusterList(clusterList)

	#calculate score for new hypotheses
	calculateScore(clusterList, False)

	targetDepthArray = [target.depth() for target in __targetList__]
	if np.std(np.array(targetDepthArray)) > 0.001:
	 	raise RuntimeError("All targets must be of equal depth") #not valid

	#maximize global (cluster vise) likelihood
	globalAssociationMatrix = np.zeros((targetDepthArray[0], len(__targetList__)), dtype = int)
	for cluster in clusterList:
		if len(cluster) == 1:
			bestHypothesis = selectBestHypothesis(__targetList__[cluster[0]])
			globalAssociationMatrix[:,cluster] = bestHypothesis[:,[0]]
		else:
			selectedMeasurements = solveOptimumAssociation(cluster)
			globalAssociationMatrix[:,cluster] = selectedMeasurements
	print("globalAssociationMatrix:\n",globalAssociationMatrix, sep = "")
	print()
	return globalAssociationMatrix

def selectBestHypothesis(target):
	def recSearchBestHypothesis(target,bestScore, bestHypothesisTrack, currentHypothesisTrack = []):
		if len(target.trackHypotheses) == 0:
			if target.cummulativeNLLR <= bestScore[0]:
				bestScore[0] = target.cummulativeNLLR
				currentHypothesisTrack.append(target.measurementNumber)
				bestHypothesisTrack[:] = currentHypothesisTrack
		else:
			for hyp in target.trackHypotheses:
				currentHypothesisTrackCpy = currentHypothesisTrack.copy()
				if target.measurementNumber is not None:
					currentHypothesisTrackCpy.append(target.measurementNumber)
				recSearchBestHypothesis(hyp, bestScore, bestHypothesisTrack, currentHypothesisTrackCpy)
	bestScore = [float('Inf')]
	bestHypothesisTrack = []
	currentHypothesisTrack = []
	recSearchBestHypothesis(target, bestScore, bestHypothesisTrack)
	return np.array(bestHypothesisTrack,dtype = int, ndmin = 2).T

def solveOptimumAssociation(cluster):
	nHypInClusterArray = getHypInCluster(cluster)
	nRealMeasurementsInCluster = getRealMeasurementsInCluster(cluster)
	(A1, measurementList) = createA1(nRealMeasurementsInCluster,sum(nHypInClusterArray), cluster)
	A2 	= createA2(len(cluster), nHypInClusterArray)
	c 	= getHypScoreInCluster(cluster)
	selectedHypotheses = solveBLP(A1,A2, c)
	selectedMeasurements = hypothesesIndices2measurementsIndices(measurementList, selectedHypotheses, A1)
	# print("Solving optimal association in cluster with targets",cluster,",   \t",
	# sum(nHypInClusterArray)," hypotheses and",nRealMeasurementsInCluster,"real measurements.",sep = " ")
	# print("nHypothesesInCluster",sum(nHypInClusterArray))
	# print("nRealMeasurementsInCluster", nRealMeasurementsInCluster)	
	# print("nTargetsInCluster", len(cluster))
	# print("nHypInClusterArray",nHypInClusterArray)
	# print("c =", c)
	# print("A1", A1, sep = "\n")
	# print("A2", A2, sep = "\n")
	# print("measurementList",measurementList)
	# print("selectedHypotheses",selectedHypotheses)
	# print("selectedMeasurements",selectedMeasurements)
	return np.array(selectedMeasurements, dtype = int, ndmin = 2).T

def hypothesesIndices2measurementsIndices(measurementList, selectedHypotheses, A1):
	measurementsIndices = []
	for hypIndex in selectedHypotheses:
		hypMeasIndices = np.nonzero(A1[:,hypIndex])[0]
		hypMeas = [measurementList[i] for i in hypMeasIndices]
		hypMeas.sort(key=lambda tup: tup[0])
		measurementsIndices.append( [meas[1] for meas in hypMeas] )
	return measurementsIndices

def createA1(nRow,nCol,cluster):
	def recActiveMeasurement(target, A1, measurementList,  activeMeasurements, hypothesisIndex):
		if len(target.trackHypotheses) == 0:
			if target.measurementNumber != 0: #we are at a real measurement
				measurement = (target.scanIndex,target.measurementNumber)
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
				if target.measurementNumber != 0 and target.measurementNumber is not None: 
					#we are on a real measurement, but not on a leaf node
					measurement = (target.scanIndex,target.measurementNumber)
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
		recActiveMeasurement(__targetList__[targetIndex],A1,measurementList,activeMeasurements,hypothesisIndex)
	return A1, measurementList

def createA2(nTargetsInCluster, nHypInClusterArray):
	A2 	= np.zeros((nTargetsInCluster,sum(nHypInClusterArray)), dtype = bool)
	colOffset = 0
	for rowIndex, nHyp in enumerate(nHypInClusterArray):
		for colIndex in range(colOffset, colOffset + nHyp):
			A2[rowIndex,colIndex]=True
		colOffset += nHyp
	return A2

def getHypScoreInCluster(cluster):
	def getTargetScore(target, scoreArray):
		if len(target.trackHypotheses) == 0:
			scoreArray.append(target.cummulativeNLLR)
		else:
			for hyp in target.trackHypotheses:
				getTargetScore(hyp, scoreArray)
	scoreArray = []
	for targetIndex in cluster:
		getTargetScore(__targetList__[targetIndex], scoreArray)
	return scoreArray

def getHypInCluster(cluster):
	def nLeafNodes(target):
		if len(target.trackHypotheses) == 0:
			return 1
		else:
			return sum(nLeafNodes(hyp) for hyp in target.trackHypotheses)
	nHypInClusterArray = np.zeros(len(cluster), dtype = int)
	for i, targetIndex in enumerate(cluster):
		nHypInTarget = nLeafNodes(__targetList__[targetIndex])
		nHypInClusterArray[i] = nHypInTarget
	return nHypInClusterArray

def getRealMeasurementsInCluster(cluster):
	def nRealMeasurements(target, measurementsInCluster):
		if target.measurementNumber != 0 and target.measurementNumber is not None:
			measurementsInCluster.add((target.scanIndex, target.measurementNumber))
		else:
			for hyp in target.trackHypotheses:
				nRealMeasurements(hyp, measurementsInCluster) 
	measurementsInCluster = set()
	for targetIndex in cluster:
		nRealMeasurements(__targetList__[targetIndex], measurementsInCluster)
	return len(measurementsInCluster)

def dfs(target, measurementHistory, h, depth = 0):
	import numpy as np
	if depth == 0: #root node
		print("Root")
		for hyp in target.trackHypotheses:
			dfs(hyp, measurementHistory, h, depth +1)
	elif len(target.trackHypotheses): #We are not at a leaf node
		for hyp in target.trackHypotheses:
			print("Midle node")
			dfs(hyp, (measurementHistory[:]).append(target.measurementNumber), h, depth+1)
	else: #we are at a leaf node
		print("Leaf node")
		mHistCopy = list(measurementHistory)
		mHistCopy.append(target.measurementNumber)
		nMeas = len(mHistCopy)
		print("nMeas", nMeas)
		h.append(mHistCopy)
		print("h",h)
		print("measurementHistory",mHistCopy)

def solveBLP(A1, A2, f):
	import numpy as np
	(nMeas, nHyp) = A1.shape
	(nTargets, _) = A2.shape
	nCost = len(f)
	if nCost != nHyp:
		raise RuntimeError("The number of costs and hypotheses must be equal")
	# print("nMeas=",nMeas, "nHyp=",nHyp, "nCost", nCost)
	prob= pulp.LpProblem("Association problem", pulp.LpMaximize)
	x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
	c = pulp.LpVariable.dicts("c", range(nHyp))
	for i in range(len(f)):
		c[i] = f[i]
	
	prob += pulp.lpSum(c[i]*x[i] for i in range(nHyp))

	for row in range(nMeas):
		prob += pulp.lpSum([ A1[row,col] * x[col] for col in range(nHyp) ]) == 1

	for row in range(nTargets):
		prob += pulp.lpSum([ A2[row,col] * x[col] for col in range(nHyp) ]) == 1

	sol = prob.solve()

	# print(prob)
	# print("Status", pulp.LpStatus[prob.status])
	selectedHypotheses = []
	for v in prob.variables():
		try:
			x = int(v.name[2:])
		except ValueError:
			continue
		# print("x_",x," \t= ", v.varValue, sep = "")
		if v.varValue == 1:
			selectedHypotheses.append(x)

	selectedHypotheses.sort()
	return selectedHypotheses

def assosiationToAdjacencyMatrix(assosiationMatrix):
	(nRow, nCol) = assosiationMatrix.shape
	nNodes = nRow + nCol
	adjacencyMatrix  = np.zeros((nNodes,nNodes),dtype=bool)
	vertices = np.nonzero(assosiationMatrix)
	adjacencyMatrix[vertices[1],vertices[0]+nCol] = True
	adjacencyMatrix[vertices[0]+nCol,vertices[1]] = True
	return adjacencyMatrix

def calculateScore(clusterList, debug):
	for clusterIndex, cluster in enumerate(clusterList):
		if debug:
			print("Calculating score for hypotheses in cluster", clusterIndex, "with target:", end= " ")
			print(*cluster, sep = ",")
		for targetIndex in cluster:
			target = __targetList__[targetIndex]
			predictedMeasurement = np.dot(C,target.predictedStateMean)
			if debug:
				print(	"\tTarget: ",targetIndex,
					"\tInit",	target.initial.position,
					"\tPred",	target.predictedPosition(),
					"\tMeas",	target.measurement,sep = "")
			for hypothesisIndex, hypothesis in enumerate(target.trackHypotheses):
				measurement = hypothesis.measurement
				measurementResidual = measurement.toarray() - predictedMeasurement
				measurementResidualLength = np.linalg.norm(measurementResidual)
				nllr = NLLR(hypothesisIndex, 
					measurement, predictedMeasurement, lambda_ex, hypothesis.residualCovariance, P_d)
				hypothesis.cummulativeNLLR = target.cummulativeNLLR + nllr
				np.set_printoptions(formatter = {'float':'{: 06.4f}'.format})
				if debug:
					print("  \t\tAlternative ",hypothesisIndex,":"
						," \tMeas", hypothesis.measurement
						," \tFilt", hypothesis.initial.position
						," \tResidual: ", measurementResidual
						," \t|R|: ", '{:06.4f}'.format(measurementResidualLength)
						," \tNLLR: ", '{: 06.4f}'.format(hypothesis.cummulativeNLLR)
						, sep = "")
			if debug:
				print()

def findClusters(assosiationMatrix):
	import numpy as np
	import scipy as sp
	adjacencyMatrix = assosiationToAdjacencyMatrix(assosiationMatrix)
	# print("Adjacency Matrix:")
	# print(adjacencyMatrix)
	# print()
	(_, nCol) = assosiationMatrix.shape
	(nClusters, labels) = sp.sparse.csgraph.connected_components(adjacencyMatrix, False)
	labels = labels[:nCol]
	clusterList = []
	for clusterIndex in range(nClusters):
		clusterList.append( (np.where( labels == clusterIndex )[0]).tolist()  )
	return clusterList

def processTarget(target, measurementList, scanIndex, usedMeasurements):
	if target.isAlive:
		if len(target.trackHypotheses)!=0:
			for hyp in target.trackHypotheses:
				processTarget(hyp, measurementList, scanIndex, usedMeasurements)
		else:
			#calculate estimated position for alive tracks
			target.kfPredict(A,Q,b, Gamma)
			#assign measurement to tracks
			target.associateMeasurements(measurementList, __sigma__, scanIndex, C, R, d, usedMeasurements)
			#plot velocity arrow and covariance ellipse
			hpf.plotVelocityArrow(target)
			hpf.plotDummyMeasurement(target)
			hpf.plotValidationRegion(target,__sigma__, C, R)
			hpf.plotDummyMeasurementIndex(scanIndex, target)