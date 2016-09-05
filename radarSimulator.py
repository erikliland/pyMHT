class SimTarget:
	def __init__(self, P0, V0, Time):
		self.initialVelocity = V0
		self.measurements = [Measurement(P0, Time)]

	def print(self):
		print("initialVelocity: (", round(self.initalVelocity.x,3) , ",\t" , round(self.Velocity.y,3) ,")", sep = "")
		print("measurments:", end = "")
		for measurement in self.measurements:
			print("\t(", round(self.Position.x,2) , ",", round(self.Position.y,2) , ")", sep="", end = "")
		print()
	def calculateAndAddNestMeasurement(self,timestep, R):
		from random import gauss
		x = self.measurements[-1].Position.x + self.initialVelocity.x * timestep + gauss(0, R[0,0])
		y = self.measurements[-1].Position.y + self.initialVelocity.y * timestep + gauss(0, R[1,1])
		self.measurements.append(Measurement(Position(x,y), self.measurements[0].Time + timestep))

def generateInitialTargets(randomSeed, numOfTargets, centerPosition, radarRange, maxSpeed, time):
	from random import seed, uniform
	from classDefinitions import Position, Velocity, Target
	from helpFunctions import pol2cart
	seed(randomSeed)
	initialList = []
	for targetIndex in range(numOfTargets):
		#Initial position and velocity
		heading  = uniform(0,360)
		distance = uniform(0,radarRange)
		px,py = pol2cart(heading,distance)
		P0 = centerPosition + Position(px,py)
		travelHeading = uniform(0,360)
		speed = uniform(0, maxSpeed)
		vx, vy = pol2cart(travelHeading, speed)
		V0 = Velocity(vx,vy)
		target = Target( P0, V0,time)
		initialList.append(target)
	return initialList

def generateScans(randomSeed, initialTargets, numOfScans, timeStep, R):
	from random import seed, uniform
	from classDefinitions import Position, Velocity, MeasurementList
	
	if len(initialTargets) == 0:
		raise ValueError("InitialTarget list must contain at least one entry")

	seed(randomSeed)
	scanTime = initialTargets[0].time
	scanList = []
	for scan in range(numOfScans):
		scanTime = scanTime + timeStep
		measurementList = MeasurementList(scanTime)
		for targetIndex, target in enumerate(initialTargets):
			tempMeasurement = target.position + target.velocity*timeStep
			measurementList.add( tempMeasurement )
		scanList.append(measurementList)
	return scanList