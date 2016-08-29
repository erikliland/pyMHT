class Position:
	def __init__(self, x, y):
		self.x = x
		self.y = y

class Velocity:
	def __init__(self, x, y):
		self.x = x #m/s
		self.y = y #m/s

class simTarget:
	def __init__(self, P0, V0):
		self.Position = P0
		self.Velocity = V0
	def Print(self):
		print("Position: (", self.Position.x, ",", self.Position.y,")\t Velocity: (",self.Velocity.x,",",self.Velocity.y,")")
	def integrateTimestep(self,timestep):
		self.Position.x += self.Velocity.x * timestep
		self.Position.y += self.Velocity.y * timestep
	def calculateNextPosition(self,timestep):
		x = self.Position.x + self.Velocity.x * timestep
		y = self.Position.y + self.Velocity.y * timestep
		return simTarget(Position(x,y),self.Velocity)

def generateMeasurementMatrix(numOfTargets, numOfScans):
	import random
	random.seed("test")
	timeStep = 2 #second
	measurmentMatrix = []
	initalList = []
	for target in range(numOfTargets):
		x0 	= random.uniform(-30, 30)
		y0 	= random.uniform(-30, 30)
		vX	= random.uniform(-2, 2)
		vY 	= random.uniform(-2, 2)
		tempTarget = simTarget( Position(x0,y0) , Velocity(vX,vY) )
		initalList.append(tempTarget)
	measurmentMatrix.append(initalList)

	for scan in range(1,numOfScans+1):
		scanList = []
		for targetIndex in range(numOfTargets):
			scanList.append( measurmentMatrix[scan-1][targetIndex].calculateNextPosition(timeStep) )
		measurmentMatrix.append(scanList)
	return measurmentMatrix
