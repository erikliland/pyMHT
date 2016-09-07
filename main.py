from classDefinitions import *
from helpFunctions import *
import radarSimulator as sim
import matplotlib.pyplot as plt
import time
import numpy as np
import tomht

initalTime = time.time()
randomSeed = "trackGen"
nTargets = 5
centerPosition = Position(1,1)
radarRange = 7.0 #meters
maxSpeed = 1 #meters/second
initialTargets = sim.generateInitialTargets(randomSeed, nTargets, centerPosition, radarRange, maxSpeed, initalTime)
initialTargets[3].velocity.x *= -1
print("Initial targets:")
print(*initialTargets, sep='\n', end = "\n\n")
fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
plotRadarOutline(centerPosition, radarRange)
plotTargetList(initialTargets)
plotVelocityArrow(initialTargets, tomht.timeStep)
nScans = 4

seed = "trackGen"
scanList = sim.generateScans(seed, initialTargets, nScans, tomht.timeStep, tomht.R0)
print("Scan list:")
print(*scanList, sep = "\n", end = "\n\n")

for initialTarget in initialTargets:
 	tomht.initiateTrack(initialTarget)



for measurementList in scanList:
	tomht.addMeasurementList(measurementList)
	#plotMeasurements(measurementList)
	#tomht.plotCovariance(tomht.__sigma__)
	break
tomht.printTargetList()
# tomht.printMeasurementAssosiation()
# tomht.printClusterList()

# print("tomht.__targetList__:")
# print(tomht.__targetList__)
# print("tomht.__clusterList__")
# print(tomht.__clusterList__)

# plt.axis("equal")
# plt.xlim((centerPosition.x-radarRange*1.05, centerPosition.x + radarRange*1.05))
# plt.ylim((centerPosition.y-radarRange*1.05, centerPosition.y + radarRange*1.05))
# plt.show(block = True)
print("Done :)")