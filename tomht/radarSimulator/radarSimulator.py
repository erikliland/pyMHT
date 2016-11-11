import numpy as np
from classDefinitions import *
import time

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
		return (repr(self.Position())+" "+
				repr(self.Velocity()) )

	def __repr__(self):
		return '({:.3e},{:.3e},{:.3e},{:.3e})'.format(*self.state)

	def Position(self):
		return Position(self.state[0],self.state[1])

	def Velocity(self):
		return Velocity(self.state[2],self.state[3])

	def calculateNextState(self, timeStep, Phi, Q, Gamma):
		w = np.random.multivariate_normal(np.zeros(2), Q)
		nextState = Phi.dot(self.state) + Gamma.dot(w.T)
		return SimTarget(nextState, self.time + timeStep)

	def positionWithNoiseAndLoss(self, H,  R, P_d = 1, p0 = None, radarRange = None):
		if np.random.uniform() < P_d:
			v = np.random.multivariate_normal(np.zeros(2), R)
			state = H.dot(self.state) + v
			return Position(state[0], state[1])
		else:
			return _generateClutter(p0,radarRange)

def generateInitialTargets(randomSeed, numOfTargets, centerPosition, radarRange, maxSpeed):
	np.random.seed(randomSeed)
	initialTime = time.time()
	initialList = []
	for targetIndex in range(numOfTargets):
		heading = np.random.uniform(0,360)
		distance= np.random.uniform(0,radarRange*0.8)
		px,py 	= _pol2cart(heading,distance)
		P0 		= centerPosition + Position(px,py)
		heading = np.random.uniform(0,360)
		speed 	= np.random.uniform(maxSpeed*0.2, maxSpeed)
		vx, vy 	= _pol2cart(heading, speed)
		V0 		= Velocity(vx,vy)
		target 	= SimTarget(np.array([px,py,vx,vy]),initialTime)
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

def simulateScans(randomSeed, simList, H, R, shuffle = True, lambda_phi = None, rRange = None, p0 = None, P_d = 1):	
	np.random.seed(randomSeed)
	area = np.pi * np.power(rRange,2)
	nClutter = int(np.floor(lambda_phi * area))
	#print("nClutter",nClutter)
	scanList = []
	for scan in simList:
		measurementList = MeasurementList(scan[0].time)
		measurementList.measurements = [target.positionWithNoiseAndLoss(H, R, P_d, p0, rRange) for target in scan]
		if (lambda_phi is not None) and (rRange is not None) and (p0 is not None):
			for i in range(int(nClutter * np.random.normal(1,0.05))):
				clutter = _generateClutter(p0, rRange)
				measurementList.measurements.append( clutter)
		scanList.append(measurementList)
	if shuffle:
		for measurementList in scanList:
			np.random.shuffle(measurementList.measurements)
	return scanList

def importFromFile(filename, **kwargs):
	startLine = kwargs.get('startLine', 0)
	initialTime = time.time()
	initialTargets = []
	simList = []
	firstPositions = None
	firstTime = None
	try:
		f = open(filename,'r')
	except:
		print("Could not open the file:", filename)
		raise
	for lineIndex, line in enumerate(f):
		lineIndex = lineIndex-startLine
		elements = line.strip().split(',')
		localTime = float(elements[0])
		globalTime = initialTime + localTime
		if lineIndex == 0:
			firstTime = float(elements[0])
			firstPositions = [Position(elements[i],elements[i+1]) for i in range(1,len(elements),2)]
		elif lineIndex > 0:
			if lineIndex == 1:
				for i,initPos in enumerate(firstPositions):
					initialTargets.append(
				SimTarget(	time = firstTime,
							position = initPos,
							velocity = (Position(elements[2*i+1],elements[2*i+2])-initPos)*(1/(localTime-firstTime)) ))

			if localTime.is_integer():
				targetList = [SimTarget(time = localTime,
										position = Position(elements[i],elements[i+1]),
										velocity = Velocity(0,0)) for i in range(1,len(elements),2)]
				simList.append( targetList )
	return initialTargets, simList

def findCenterPositionAndRange(simList):
	xMin = float('Inf')
	yMin = float('Inf')
	xMax =-float('Inf')
	yMax =-float('Inf')
	for sim in simList:
		for simTarget in sim:
			xMin = simTarget.state[0] if simTarget.state[0] < xMin else xMin
			yMin = simTarget.state[1] if simTarget.state[1] < yMin else yMin
			xMax = simTarget.state[0] if simTarget.state[0] > xMax else xMax
			yMax = simTarget.state[1] if simTarget.state[1] > yMax else yMax
	p0 = Position( xMin+(xMax-xMin)/2 , yMin + (yMax-yMin)/2 )
	R = max(xMax-xMin, yMax-yMin)
	return p0,R

def _generateClutter(centerPosition, radarRange):
	heading = np.random.uniform(0,360)
	distance= np.random.uniform(0,radarRange)
	px,py 	= _pol2cart(heading,distance)
	return centerPosition + Position(px,py)

def _pol2cart(bearingDEG,distance):
	angleDEG = 90 - bearingDEG
	angleRAD = np.deg2rad(angleDEG)
	x = distance * np.cos(angleRAD)
	y = distance * np.sin(angleRAD)
	return [x,y]
