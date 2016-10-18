import numpy as np
import helpFunctions as hpf
from classDefinitions import *

class SimTarget:
	def __init__(self, *args, **kwargs):
		p = kwargs.get('position')
		v = kwargs.get('velocity')
		t = kwargs.get('time')
		if p and v and t is not None:
			self.state = np.array([p.x,p.y,v.x,v.y])
			self.time = t
		elif len(args) == 2:
			self.state = args[0]
			self.time  = args[1]
		else:
			print("Invalid arguments to SimTarget")

	def __str__(self):
		return (repr(Position(self.state[0],self.state[1]))+" "+
				repr(Velocity(self.state[2],self.state[3])) )

	__repr__ = __str__

	def calculateNextState(self, timeStep, Phi, Q, Gamma):
		w = np.random.multivariate_normal(np.zeros(2), Q)
		nextState = Phi.dot(self.state) + Gamma.dot(w.T)
		return SimTarget(nextState, self.time + timeStep)

	def positionWithNoise(self, H,  R):
		v = np.random.multivariate_normal(np.zeros(2), R)
		state = H.dot(self.state) + v
		return Position(state[0], state[1])

def generateInitialTargets(randomSeed, numOfTargets, centerPosition, radarRange, maxSpeed):
	import time
	np.random.seed(randomSeed)
	initalTime = time.time()
	initialList = []
	for targetIndex in range(numOfTargets):
		heading = np.random.uniform(0,360)
		distance= np.random.uniform(0,radarRange*0.8)
		px,py 	= hpf.pol2cart(heading,distance)
		P0 		= centerPosition + Position(px,py)
		heading = np.random.uniform(0,360)
		speed 	= np.random.uniform(maxSpeed*0.2, maxSpeed)
		vx, vy 	= hpf.pol2cart(heading, speed)
		V0 		= Velocity(vx,vy)
		target 	= SimTarget(np.array([px,py,vx,vy]),initalTime)
		initialList.append(target)
	return initialList

def simulateTargets(randomSeed, initialTargets, numOfSteps, timeStep, Phi, Q, Gamma):
	np.random.seed(randomSeed)
	simList = []
	simList.append(initialTargets)
	for scan in range(numOfSteps):
		targetList = [target.calculateNextState(timeStep, Phi, Q, Gamma) for target in simList[-1]]
		simList.append(targetList)
	simList.pop(0)
	return simList

def simulateScans(randomSeed, simList, H, R, shuffle = True):	
	np.random.seed(randomSeed)
	scanList = []
	for scan in simList:
		measurementList = MeasurementList(scan[0].time)
		measurementList.measurements = [target.positionWithNoise(H, R) for target in scan]
		scanList.append(measurementList)
	if shuffle:
		for measurementList in scanList:
			np.random.shuffle(measurementList.measurements)
	return scanList