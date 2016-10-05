import numpy as np
import helpFunctions as hpf
from classDefinitions import *

class SimTarget:
	def __init__(self, *args, **kwargs):
		p = kwargs.get('position')
		v = kwargs.get('velocity')
		if p and v is not None:
			self.state = np.array([p.x,p.y,v.x,v.y])
		elif len(args) == 1:
			self.state = args[0]
		else:
			error("Invalid arguments to to SimTarget")

	def __str__(self):
		return (repr(Position(self.state[0],self.state[1]))+" "+
				repr(Velocity(self.state[2],self.state[3])) )

	__repr__ = __str__

	def calculateNextState(self, Phi, Q, Gamma):
		w = np.random.multivariate_normal(np.zeros(2), Q)
		nextState = Phi.dot(self.state) + Gamma.dot(w.T)
		return SimTarget(nextState)

	def positionWithNoise(self, H,  R):
		v = np.random.multivariate_normal(np.zeros(2), R)
		state = H.dot(self.state) + v
		return Position(state[0], state[1])

def generateInitialTargets(randomSeed, numOfTargets, centerPosition, radarRange, maxSpeed, time):
	np.random.seed(randomSeed)
	initialList = []
	for targetIndex in range(numOfTargets):
		#Initial position and velocity
		heading = np.random.uniform(0,360)
		distance= np.random.uniform(0,radarRange*0.8)
		px,py 	= hpf.pol2cart(heading,distance)
		P0 		= centerPosition + Position(px,py)
		heading = np.random.uniform(0,360)
		speed 	= np.random.uniform(maxSpeed*0.2, maxSpeed)
		vx, vy 	= hpf.pol2cart(heading, speed)
		V0 		= Velocity(vx,vy)
		target 	= initialTarget( P0, V0,time)
		initialList.append(target)
	return initialList

def generateScans(randomSeed, initialTargets, numOfScans, timeStep, Phi, H, Gamma, Q, R):	
	np.random.seed(randomSeed)
	simList = []
	initialMeasurements = [SimTarget(position = initialTarget.position, velocity = initialTarget.velocity) for initialTarget in initialTargets]
	simList.append(initialMeasurements)
	for scan in range(numOfScans):
		targetList = [target.calculateNextState(Phi, Q, Gamma) for target in simList[-1]]
		simList.append(targetList)
	simList.pop(0)

	print("Simulated list:")
	print(*simList, sep = "\n", end = "\n\n")

	scanTime = initialTargets[0].time
	scanList = []
	for scan in simList:
		scanTime += timeStep
		measurementList = MeasurementList(scanTime)
		measurementList.measurements = [target.positionWithNoise(H, R) for target in scan]
		scanList.append(measurementList)

	# for measurementList in scanList:
	# 	np.random.shuffle(measurementList.measurements)
	return scanList