##Initiation starting
import numpy as np
#State space model
C 		= np.array([[1, 0, 0, 0],
					[0, 1, 0, 0]]) #also known as "H"
Gamma 	= np.array([[0,0],
					[0,0],
					[1,0],
					[0,1]]) #Disturbance matrix (only velocity)
P0 		= np.eye(5) * 1e-5
q 		= 0.04
__trackList__ = []
__clusterList__ = []
__lastMeasurementTime__ = -1.0
__sigma__ = 2
__scanHistory__ = []
##Initiation finished

def Phi(T):
	from numpy import array
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])

def Q(q,T):
	from numpy import eye
	return np.eye(2)*q*T

def initiateTrack(target):
	state = np.array([target.position.x,target.position.y, target.velocity.x,target.velocity.y])
	tempTrack = Track(state, target.time)
	__trackList__.append(tempTrack)
	__clusterList__.append(len(__trackList__))

def addMeasurementList(measurementList):
	__scanHistory__.append(measurementList)
	now = measurementList.time
	#calculate estimated position for alive tracks
	for track in __trackList__:
		if track.isAlive:
			track.calculateAprioriState(now)
			track.calculateAprioriCovariance
	
	#assign measurement to tracks
	for track in __trackList__:
		if track.isAlive:
			track.assosiateMeasurements(measurementList.measurements)
	
	#re-cluster / merge-split clusters
	reCluster()

	#process each cluster (in parallel...?)
	
	
	#store updated result in track list

def reCluster():
	__clusterList__.clear()
	assignedTracs = []
	for trackIndex, track in enumerate(__trackList__):
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

def printClusterList():
	print("Cluster list:")
	# print(*__clusterList__, sep = "", end = "\n\n")
	for clusterIndex, cluster in enumerate(__clusterList__):
		print("Cluster ", clusterIndex, " contains track:\t", cluster, sep ="", end = "\n")
	print()

def printMeasurementAssosiation():
	print("Track-measurement assosiation:")
	for trackIndex, track in enumerate(__trackList__):
		if track.isAlive:
			print("Track ",trackIndex, " is assosiated with measurements:\t",sep = "", end = "")
			print(*track.assosiatedMeasurements, sep = "; ")
	print()

def plotCovariance(sigma):
		from helpFunctions import plotCovarianceEllipse
		for track in __trackList__:
			if not track.isAlive:
				break
			plotCovarianceEllipse(track.aprioriEstimatedCovariance, track.aprioriEstimatedState[0:2], sigma)

class Track:
	def __init__(self, state, time):
		from numpy import array, eye
		self.time = time
		self.isAlive = True
		self.assosiatedMeasurements = [] #indecies to measurements in measurementList
		self.state = state
		self.aprioriEstimatedState = self.state
		self.aprioriEstimatedCovariance = eye(4)*0.04
		self.kalmanGain = 0
		self.aposterioriEstimatedState = self.state
		self.aposterioriEstimatedCovariance = eye(2) * 0.04
	def __str__(self):
		return str(self.state)

	__repr__ = __str__

	def assosiateMeasurements(self, measurements):
		foundMeasurement = False
		measurementCandidates = []
		for measurementIndex, measurement in enumerate(measurements):
			if self.measurementIsInsideErrorEllipse(measurement, __sigma__):
				measurementCandidates.append(measurementIndex)
				foundMeasurement = True
		if not foundMeasurement:
			measurementCandidates.append(-1)
		self.assosiatedMeasurements.append(measurementCandidates)

	def calculateAprioriState(self, now):
		from numpy import dot
		T = now - self.time
		self.aprioriEstimatedState = dot(Phi(T),self.aprioriEstimatedState)

	def calculateAprioriCovariance(self):
		from numpy import dot, transpose
		T = now - self.time
		A = Phi(T)
		Q = Q(T)
		self.aprioriEstimatedCovariance = A.dot(self.aprioriEstimatedCovariance).dot(A.transpose()) + Gamma.dot(Q).dot(Gamma.transpose())

	def calculateAposterioriEstimatedCovariance(self, H, R):
		from numpy import dot
		from numpy.linalg import inv
		self.aposterioriEstimatedCovariance=self.aprioriEstimatedCovariance-self.aprioriEstimatedCovariance.dot(H.transpose()).dot(inv(H.dot(self.aprioriEstimatedCovariance).dot(H.transpose())+R)).dot(H.dot(self.aprioriEstimatedCovariance))

	def calculateKalmanGain(self, H, R):
		from numpy import dot
		from numpy.linalg import inv
		self.kalmanGain = self.aposterioriEstimatedCovariance.dot(H.transpose()).dot(inv(R))

	def calculateAposterioriStateEstimate(self,z, H):
		self.aposterioriEstimatedState = self.aprioriEstimatedState + self.kalmanGain * ( z - H.dot(self.aprioriEstimatedState) )


	def printAprioriState(self):
		print("Apriori state:\tP(", round(self.aprioriEstimatedState[0],3), ",", round(self.aprioriEstimatedState[1],3), 
				")  \tV(", round(self.aprioriEstimatedState[2],3), ",", round(self.aprioriEstimatedState[3],3),")", sep = "")

	def measurementIsInsideErrorEllipse(self, measurement, sigma):
		from numpy.linalg import eig
		from numpy import sqrt, rad2deg, arccos, cos, sin, power

		lambda_, v = eig(self.aprioriEstimatedCovariance[0:2,0:2])
		lambda_ = sqrt(lambda_)
		deltaX = measurement.x - self.aprioriEstimatedState[0]
		deltaY = measurement.y - self.aprioriEstimatedState[1]
		a = lambda_[0]*sigma
		b = lambda_[1]*sigma
		angle = arccos(v[0,0])
		sum = power( cos(angle)*deltaX+sin(angle)*deltaY,2 )/power(a,2)  + power( sin(angle)*deltaX-cos(angle)*deltaY,2)/power(b,2)
		if sum <= 1:
			return True
		else:
			return False
