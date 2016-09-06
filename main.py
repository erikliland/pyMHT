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
timeStep = 1 #second
fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
plotRadarOutline(centerPosition, radarRange)
plotTargetList(initialTargets)
plotVelocityArrow(initialTargets, timeStep)

nScans = 4
r 		= 0.04 #Measurement variance
R 		= np.eye(2)*r 	#Measurement covariance
seed = "trackGen"
scanList = sim.generateScans(seed, initialTargets, nScans, timeStep, R)
print("Scan list:")
print(*scanList, sep = "\n", end = "\n\n")

for target in initialTargets:
	tomht.initiateTrack(target)

for measurementList in scanList:
	tomht.addMeasurementList(measurementList)
	plotMeasurements(measurementList)
	tomht.plotCovariance(tomht.__sigma__)
	break

tomht.printMeasurementAssosiation()
tomht.printClusterList()

# print("tomht.__trackList__:")
# print(tomht.__trackList__)
# print("tomht.__clusterList__")
# print(tomht.__clusterList__)

plt.axis("equal")
plt.xlim((centerPosition.x-radarRange*1.05, centerPosition.x + radarRange*1.05))
plt.ylim((centerPosition.y-radarRange*1.05, centerPosition.y + radarRange*1.05))
# plt.show(block = True)
print("Done :)")