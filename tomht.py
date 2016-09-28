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

def Phi(T):
	from numpy import array
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])

#State space model
timeStep = 1 #second
A 		= Phi(timeStep) #transition matrix

b 		= np.zeros(4) 	#transition offsets

C 		= np.array([[1, 0, 0, 0],
					[0, 1, 0, 0]]) #also known as "H"

d 		= np.zeros(2)	#observation offsets

Gamma 	= np.array([[0,0],
					[0,0],
					[1,0],
					[0,1]]) #Disturbance matrix (only velocity)
P0 		= np.eye(4) * 1e-5
q 		= 0.04
P_d 	= 0.8
r 		= 0.05 			#Measurement variance
q 		= 0.022	 		#System variance
R0 		= np.eye(2) * r #Initial Measurement/observation covariance
Q0		= np.eye(4) * q	#Initial transition/system covariance
lambda_ex = 1 			#Spatial density of the extraneous measurements (expected number per volume in scan k)

__targetList = []
__clusterList__ = []
__lastMeasurementTime__ = -1.0
__sigma__ = 2
__scanHistory = []
##Initiation finished

def Q(q,T):
	from numpy import eye
	return np.eye(2)*q*T

def NLLR(hypothesisIndex, measurement,predictedMeasurement,lambda_ex,covariance,P_d):
	from numpy import dot, transpose, log, pi, power
	from numpy.linalg import inv, det
	measurementResidual = measurement.toarray() - predictedMeasurement
	if hypothesisIndex == 0:
		return -log(1-P_d)
	else:
		return (	0.5*(measurementResidual.transpose().dot(inv(covariance)).dot(measurementResidual))
					+ log((lambda_ex*power(det(2*pi*covariance),0.5))/P_d) 	)

def getKalmanFilterInitData(initialTarget):
	import numpy as np
	return Bunch(	transitionMatrix 				= A,
					observationMatrix 				= C,
					initialTransitionCovariance 	= Q0,
					initialObservationCovariance 	= R0,
					transitionOffsets 				= b,
					observationOffsets 				= d,
					initialStateMean 				= initialTarget.state(), 
					initialStateCovariance 			= P0,
					randomState 					= 0)

def initiateTrack(initialTarget):
	from classDefinitions import Target
	currentScanIndex = len(__scanHistory)
	kfData = getKalmanFilterInitData(initialTarget)
	tempTarget = Target(initialTarget, currentScanIndex, None, None, kfData)
	__targetList.append(tempTarget)
	hpf.plotInitialTargetIndex(initialTarget, len(__targetList)-1)

def addMeasurementList(measurementList):
	import numpy as np
	__scanHistory.append(measurementList)
	scanIndex = len(__scanHistory)
	print("#"*150)
	print("Adding scan index:", scanIndex)
	hpf.plotMeasurements(measurementList)
	hpf.plotMeasurementIndecies(scanIndex, measurementList.measurements)
	nMeas = len(measurementList.measurements)
	nTargets = len(__targetList)
	associationMatrix = np.zeros((nMeas, nTargets),dtype = np.bool)
	for targetIndex, target in enumerate(__targetList):
		processTarget(target, measurementList, scanIndex, associationMatrix[:,targetIndex])
		hpf.printMeasurementAssociation(targetIndex, target)
	
	print("AssosiationMatrix")
	print(associationMatrix)
	print()

	#cluster / merge-split clusters
	clusterList = findClusters(associationMatrix)
	hpf.printClusterList(clusterList)

	#process each cluster (generate hypothesis and calculate score)
	processClusters(clusterList)

	#maximize global (cluster vise) likelihood
	for clusterIndex, cluster in enumerate(clusterList):
		association = solveOptimumAssociation(cluster, associationMatrix)
		break

	#store updated result in track list


def solveOptimumAssociation(cluster, associationMatrix):
	print("Cluster:")
	print(cluster)
	print("Association Matrix")
	print(associationMatrix)
	clusterAssociationMatrix = associationMatrix[:,cluster]
	activeMeasurements = np.nonzero(clusterAssociationMatrix)
	uniqueActiveMeasurements = np.unique(activeMeasurements[0])
	reducedAssociationMatrix = clusterAssociationMatrix[uniqueActiveMeasurements,:]
	print("Active measurements")
	print(activeMeasurements)
	print("ClusterAssociationMatrix")
	print(clusterAssociationMatrix)
	print("reducedAssociationMatrix")
	print(reducedAssociationMatrix)
	##### Need to add dummy measurements somewhere...
	# Aeq = np.array([[1,1,1,0],
	# 				[0,0,0,1]])
	# f 	= np.array([1.2,0.6,1.5,1.1]) 
	# x 	= solveBLP(Aeq, f)
	# print("Aeq:\n", Aeq, sep="")
	# print("f:", f)
	# print("x:", x)
	

def solveBLP(Aeq, f):
	import numpy as np
	(nMeas, nHyp) = Aeq.shape
	nCost = len(f)
	if nCost != nHyp:
		error("The number of costs and hypotheses must be equal")
	# print("nMeas=",nMeas, "nHyp=",nHyp, "nCost", nCost)
	prob= pulp.LpProblem("Association problem", pulp.LpMaximize)
	x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
	c = pulp.LpVariable.dicts("c", range(nHyp))
	for i in range(len(f)):
		c[i] = f[i]
	
	prob += pulp.lpSum(c[i]*x[i] for i in range(nHyp))

	for row in range(nMeas):
		prob += pulp.lpSum([ Aeq[row,col] * x[col] for col in range(nHyp) ]) == 1

	sol = prob.solve()

	# print(prob)
	# print("Status", pulp.LpStatus[prob.status])
	selectedHypotheses = []
	for i, v in enumerate(prob.variables()):
		# print(v.name, "=", v.varValue)
		if v.varValue == 1:
			selectedHypotheses.append(i)
	return selectedHypotheses

def assosiationToAdjacencyMatrix(assosiationMatrix):
	import numpy as np
	(nRow, nCol) = assosiationMatrix.shape
	nNodes = nRow + nCol
	adjacencyMatrix  = np.zeros((nNodes,nNodes),dtype=bool)
	vertices = np.nonzero(assosiationMatrix)
	adjacencyMatrix[vertices[1],vertices[0]+nCol] = True
	adjacencyMatrix[vertices[0]+nCol,vertices[1]] = True
	return adjacencyMatrix

def processClusters(clusterList):
	from numpy import dot, transpose
	import numpy as np
	from numpy.linalg import norm
	for clusterIndex, cluster in enumerate(clusterList):
		print("Processing cluster", clusterIndex, "with target:", end= " ")
		print(*cluster, sep = ",")
		for targetIndex in cluster:
			target = __targetList[targetIndex]
			predictedMeasurement = np.dot(C,target.predictedStateMean)
			print("\tTarget: ", targetIndex,
				"\tInit",target.initial.position,
				"\tPred",target.predictedPosition(),
				"\tMeas",target.measurement,sep = "")
			for hypothesisIndex, hypothesis in enumerate(target.trackHypotheses):
				measurement = hypothesis.measurement
				measurementResidual = measurement.toarray() - predictedMeasurement
				measurementResidualLength = norm(measurementResidual)
				(K, state, cov ) = pk._filter_correct(C, 
					R0, d, target.predictedStateMean, target.predictedStateCovariance, measurement.toarray())
				correctedMeasurementCovariance = C.dot(cov).dot(C.transpose())
				nllr = NLLR(hypothesisIndex, 
					measurement, predictedMeasurement, lambda_ex, correctedMeasurementCovariance, P_d)
				hypothesis.cumulativeNLLR = target.cummulativeNLLR + nllr
				np.set_printoptions(formatter = {'float':'{: 06.4f}'.format})
				print("  \t\tAlternative ",hypothesisIndex,":"
						," \tMeas", hypothesis.measurement
						," \tFilt", hypothesis.initial.position
						," \tResidual: ", measurementResidual
						," \t|R|: ", '{:06.4f}'.format(measurementResidualLength)
						," \tNLLR: ", '{: 06.4f}'.format(nllr)
						, sep = "")
			print()

def findClusters(assosiationMatrix):
	import numpy as np
	import scipy as sp
	adjacencyMatrix = assosiationToAdjacencyMatrix(assosiationMatrix)
	print("Adjacency Matrix:")
	print(adjacencyMatrix)
	print()
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
			target.kfPredict(A,Q0,b)
			#assign measurement to tracks
			target.associateMeasurements(measurementList, __sigma__, scanIndex, C, R0, d, usedMeasurements)
			#plot velocity arrow and covariance ellipse
			hpf.plotVelocityArrow(target)
			hpf.plotDummyMeasurement(target)
			hpf.plotCovariance(target,__sigma__, C)
			hpf.plotDummyMeasurementIndex(scanIndex, target)