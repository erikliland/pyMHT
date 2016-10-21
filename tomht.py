"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Authumn 2016
========================================================================================
"""
import numpy as np
import scipy as sp
import helpFunctions as hpf
import pulp
from classDefinitions import Target, Position, Velocity

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
P_d 	= 0.9						#Probability of detection
p 		= np.power(1e-2,1)			#Initial systen state variance
P0 		= np.diag([p,p,p,p])		#Initial state covariance
r		= np.power(1e-3,1)			#Measurement variance
q 		= np.power(2e-2,1)			#Velocity variance variance
R 		= np.eye(2) * r 			#Measurement/observation covariance
Q		= np.eye(2) * q * T 		#Transition/system covariance (process noise)
lambda_phi 	= 0.05					#Expected number of false measurements per unit 
									# volume of the measurement space per scan
lambda_nu 	= 0.001					#Expected number of new targets per unit volume 
									# of the measurement space per scan
lambda_ex 	= lambda_phi+lambda_nu 	#Spatial density of the extraneous measurements
sigma 		= 3						#Need to be changed to conficence
windowSize 	= 3						#Number of  timesteps to tail (N-scan)

__targetList__ 	= []
__associatedMeasurements__ = []
__scanHistory__ = []

def initiateTarget(newTarget):
	target = Target(	time = newTarget.time, 
						scanNumber =  len(__scanHistory__),
						state = newTarget.state, 
						covariance = P0)
	__targetList__.append(target)
	__associatedMeasurements__.append( set() )
	target.plotInitial(len(__targetList__)-1)

def addMeasurementList(measurementList):
	__scanHistory__.append(measurementList)
	scanNumber = len(__scanHistory__)
	print("#"*150,"\nAdding scan number:", scanNumber)
	nMeas = len(measurementList.measurements)
	nTargets = len(__targetList__)
	for targetIndex, target in enumerate(__targetList__):
		#estimate, gate and score new measurement
		_processNewMeasurement(target, measurementList, scanNumber, __associatedMeasurements__[targetIndex])

	# print(*__associatedMeasurements__, sep = "\n", end = "\n\n")
	#--Cluster targets--
	clusterList = _findClustersFromSets()
	# hpf.printClusterList(clusterList)

	#--Maximize global (cluster vise) likelihood--
	globalHypotheses = np.empty(len(__targetList__),dtype = np.dtype(object))
	for cluster in clusterList:
		if len(cluster) == 1:
			# _pruneSmilarState(__targetList__[cluster[0]], 0.1)
			globalHypotheses[cluster] = _selectBestHypothesis(__targetList__[cluster[0]])
		else:
			globalHypotheses[cluster] = _solveOptimumAssociation(cluster)
	return globalHypotheses

def _pruneSmilarState(target, errorNormLimit):
	from scipy.special import binom
	nHyp = len(target.trackHypotheses)
	nDelta = int(binom(nHyp,2))
	deltaX = np.zeros([4,nDelta])
	hypotheses = target.trackHypotheses[1:]
	done = set()
	for a in target.trackHypotheses[:-1]:
		for b in hypotheses:
			if a != b:
				targetID = (a.measurementNumber,b.measurementNumber)
				if targetID not in done:
					deltaX[:,len(done)] = (a.getState() - b.getState())
					done.add( targetID )
		hypotheses.pop(0)
	for col in range(nDelta):
		errorNorm = np.linalg.norm(deltaX[:,col])
		print(errorNorm)
		if errorNorm < errorNormLimit:
			pass

def _selectBestHypothesis(target):
	def recSearchBestHypothesis(target,bestScore, bestHypothesis):
		if len(target.trackHypotheses) == 0:
			if target.cummulativeNLLR <= bestScore[0]:
				bestScore[0] = target.cummulativeNLLR
				bestHypothesis[0] = target
		else:
			for hyp in target.trackHypotheses:
				recSearchBestHypothesis(hyp, bestScore, bestHypothesis)
	bestScore = [float('Inf')]
	bestHypothesis = [None]
	recSearchBestHypothesis(target, bestScore, bestHypothesis)
	return np.array(bestHypothesis)

def _solveOptimumAssociation(cluster):
	nHypInClusterArray = _getHypInCluster(cluster)
	nRealMeasurementsInCluster= len(set.union(*[__associatedMeasurements__[i] for i in cluster]))
	(A1, measurementList) = _createA1(nRealMeasurementsInCluster,sum(nHypInClusterArray), cluster)
	A2 	= _createA2(len(cluster), nHypInClusterArray)
	C 	= _createC(cluster)
	selectedHypotheses = _solveBLP(A1,A2, C)
	selectedNodes = _hypotheses2Nodes(selectedHypotheses,cluster)
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
	#return np.array(selectedMeasurements, dtype = int, ndmin = 2).T
	return np.array(selectedNodes)

def _hypotheses2Nodes(selectedHypotheses, cluster):
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
		recDFS(__targetList__[targetIndex], selectedHypotheses, nodeList, counter)
	return nodeList

def _createA1(nRow,nCol,cluster):
	def recActiveMeasurement(target, A1, measurementList,  activeMeasurements, hypothesisIndex):
		if len(target.trackHypotheses) == 0:
			if target.measurementNumber != 0: #we are at a real measurement
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
				if target.measurementNumber != 0 and target.measurementNumber is not None: 
					#we are on a real measurement, but not on a leaf node
					measurement = (target.scanNumber,target.measurementNumber)
					try:
						measurementIndex = measurementList.index(measurement)
					except ValueError:
						measurementList.append(measurement)
						measurementIndex = len(measurementList) -1
					# print("measurementIndex",measurementIndex)
					activeMeasurementsCpy[measurementIndex] = True
				recActiveMeasurement(hyp, A1, measurementList, activeMeasurementsCpy, hypothesisIndex)

	A1 	= np.zeros((nRow,nCol), dtype = bool)
	# print("nRow", nRow)
	activeMeasurements = np.zeros(nRow, dtype = bool)
	measurementList = []
	hypothesisIndex = [0]
	#TODO: http://stackoverflow.com/questions/15148496/python-passing-an-integer-by-reference
	for targetIndex in cluster:
		recActiveMeasurement(__targetList__[targetIndex],A1,measurementList,activeMeasurements,hypothesisIndex)
	return A1, measurementList

def _createA2(nTargetsInCluster, nHypInClusterArray):
	A2 	= np.zeros((nTargetsInCluster,sum(nHypInClusterArray)), dtype = bool)
	colOffset = 0
	for rowIndex, nHyp in enumerate(nHypInClusterArray):
		for colIndex in range(colOffset, colOffset + nHyp):
			A2[rowIndex,colIndex]=True
		colOffset += nHyp
	return A2

def _createC(cluster):
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

def _getHypInCluster(cluster):
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

def _solveBLP(A1, A2, f):
	(nMeas, nHyp) = A1.shape
	(nTargets, _) = A2.shape
	nCost = len(f)
	if nCost != nHyp:
		raise RuntimeError("The number of costs and hypotheses must be equal")
	# print("nMeas=",nMeas, "nHyp=",nHyp, "nCost", nCost)
	prob = pulp.LpProblem("Association problem", pulp.LpMinimize)
	x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
	c = pulp.LpVariable.dicts("c", range(nHyp))
	for i in range(len(f)):
		c[i] = f[i]
	
	prob += pulp.lpSum(c[i]*x[i] for i in range(nHyp))

	for row in range(nMeas):
		prob += pulp.lpSum([ A1[row,col] * x[col] for col in range(nHyp) ]) <= 1

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
		if v.varValue == 1:
			selectedHypotheses.append(x)
	selectedHypotheses.sort()
	return selectedHypotheses

def _findClustersFromSets():
	superSet = set() #TODO: This should be done a more elegant way!
	for targetIndex, targetSet in enumerate(__associatedMeasurements__):
		superSet |= targetSet
	nTargets = len(__associatedMeasurements__)
	nNodes = nTargets + len(superSet)
	adjacencyMatrix  = np.zeros((nNodes,nNodes),dtype=bool)
	for targetIndex, targetSet in enumerate(__associatedMeasurements__):
		for measurementIndex, measurement in enumerate(superSet):
			adjacencyMatrix[targetIndex,measurementIndex+nTargets] = (measurement in targetSet)
	# print("Adjacency Matrix2:\n", adjacencyMatrix.astype(dtype = int), sep = "", end = "\n\n")
	(nClusters, labels) = sp.sparse.csgraph.connected_components(adjacencyMatrix)
	return [np.where(labels[:nTargets]==clusterIndex)[0].tolist() for clusterIndex in range(nClusters)]

def _processNewMeasurement(target, measurementList, scanNumber, associatedMeasurements):
	if len(target.trackHypotheses) == 0:
		target.predictMeasurement(Phi(measurementList.time-target.time),Q,b, Gamma, C, R)
		target.gateAndCreateNewHypotheses(measurementList, sigma, P_d, lambda_ex, scanNumber, C, R, d, associatedMeasurements)
	else:
		for hyp in target.trackHypotheses:
			_processNewMeasurement(hyp, measurementList, scanNumber,associatedMeasurements)