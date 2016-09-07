"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Authumn 2016
=========================================================================================
"""
##Initiation starting
import pykalman.standard as pk
import numpy as np
from pykalman.utils import Bunch


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
r 		= 0.04 			#Measurement variance
R0 		= np.eye(2)*r 	#Initial Measurement/observation covariance
Q0		= np.eye(4) * 1	#Initial transition covariance
lambda_ex = 1 #Spatial density of the extraneous measurements (expected number per volume in scan k)

__targetList__ = []
__clusterList__ = []
__lastMeasurementTime__ = -1.0
__sigma__ = 2
__scanHistory__ = []
##Initiation finished

def Q(q,T):
	from numpy import eye
	return np.eye(2)*q*T

def NLLR(measurement,predictedMeasurement,lambda_ex,covariance,P_d):
	from numpy import dot, transpose, log, pi, power
	from numpy.linalg import inv, det
	measurementResidual = measurement.array() - predictedMeasurement
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
	currentScanIndex = len(__scanHistory__)
	kfData = getKalmanFilterInitData(initialTarget)
	tempTarget = Target(initialTarget, currentScanIndex, kfData)
	__targetList__.append(tempTarget)

def addMeasurementList(measurementList):
	__scanHistory__.append(measurementList)
	scanIndex = len(__scanHistory__)
	for target in __targetList__:
		if target.isAlive:
			#calculate estimated position for alive tracks
			target.kfPredict(A,Q0,b)
			#assign measurement to tracks
			target.assosiateMeasurements(measurementList.measurements, __sigma__, scanIndex, measurementList.time)
	
	return
	#re-cluster / merge-split clusters
	reCluster()
	
	
	#process each cluster (in parallel...?)[generate hypothesis and calculate score]
	for clusterIndex, cluster in enumerate(__clusterList__):
		print("Processing cluster", clusterIndex, "with tracks:", end= " ")
		print(*cluster, sep = ",")
		processCluster(cluster)
		break
	print()


	#store updated result in track list

def processCluster(cluster):
	from numpy import dot
	for trackIndex in cluster:
		track = __targetList__[trackIndex]
		print("Track:", trackIndex)
		print("Initial position:", track.initialState[0:2])
		predictedMeasurement = np.dot(C,track.aprioriEstimatedState)
		print("Predicted measurement:", predictedMeasurement)
		index = 1
		for measurementIndex in track.assosiatedMeasurements[-1]:
			measurement = __scanHistory__[-1].measurements[measurementIndex]
			measurementResidual = measurement.array() - predictedMeasurement
			nllr = NLLR(measurement, predictedMeasurement, lambda_ex, track.aprioriEstimatedCovariance[0:2,0:2], P_d)
			print("Alternative ",index,": ",measurement, "\tResidual: ", measurementResidual, "\tNLLR: ", nllr, sep = "")	
			index += 1
		print()
	pass

def reCluster():
	__clusterList__.clear()
	assignedTracs = []
	for trackIndex, track in enumerate(__targetList__):
		if not track.isAlive:
			break
		nMeasurement = len(track.assosiatedMeasurements[-1])
		if nMeasurement == 0:
			raise RuntimeError("All track must have at least one measurement at each time!")
		elif nMeasurement == 1:
			if trackIndex not in assignedTracs:
				__clusterList__.append([trackIndex])
				assignedTracs.append(trackIndex)
			else:
				raise RuntimeError("Multiple tracks have been assigned the same measurement without doubble assignment")
		elif nMeasurement > 1:
			alreadyAssignedToCluster = False
			for measurement in track.assosiatedMeasurements:
				if trackIndex in assignedTracs:
					alreadyAssignedToCluster = True
					break
			if not alreadyAssignedToCluster:
				tempList = []
				for measurement in track.assosiatedMeasurements[-1]:
					tempList.append(measurement)
					assignedTracs.append(measurement)
				__clusterList__.append(tempList)
		else:
			raise RuntimeError("Number of assosiated measurements can't be negative!")

def printTargetList():
	print("TargetList:")
	print(*__targetList__, sep = "\n")

def printClusterList():
	print("Cluster list:")
	# print(*__clusterList__, sep = "", end = "\n\n")
	for clusterIndex, cluster in enumerate(__clusterList__):
		print("Cluster ", clusterIndex, " contains track:\t", cluster, sep ="", end = "\n")
	print()

def printMeasurementAssosiation():
	print("Track-measurement assosiation:")
	for trackIndex, track in enumerate(__targetList__):
		if track.isAlive:
			print("Track ",trackIndex, " is assosiated with measurements:\t",sep = "", end = "")
			print(*track.assosiatedMeasurements, sep = "; ")
	print()

def plotCovariance(sigma):
		from helpFunctions import plotCovarianceEllipse
		for track in __targetList__:
			if not track.isAlive:
				break
			plotCovarianceEllipse(track.aprioriEstimatedCovariance, track.aprioriEstimatedState[0:2], sigma)
