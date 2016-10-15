from classDefinitions import * 
from helpFunctions import *
import radarSimulator as sim
import matplotlib.pyplot as plt
import time
import numpy as np
import tomht
import os

initalTime = time.time()
randomSeed = 5446
nTargets = 4
centerPosition = Position(1,1)
radarRange = 10.0 #meters
maxSpeed = 2 #meters/second
initialTargets = sim.generateInitialTargets(randomSeed, 
	nTargets, centerPosition, radarRange, maxSpeed, initalTime)
initialTargets[3].velocity.y *= -1
initialTargets[3].velocity *= 0.3
initialTargets[1].velocity *= 1.5
initialTargets[1].velocity.x *= -1.5
initialTargets[1].velocity *= 1.05
print("Initial targets:")
print(*initialTargets, sep='\n', end = "\n\n")
fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
plotRadarOutline(centerPosition, radarRange)
plotTargetList(initialTargets)
nScans = 5

scanList = sim.generateScans(randomSeed, initialTargets, nScans, tomht.timeStep,
			 tomht.Phi(tomht.timeStep), tomht.C, tomht.Gamma, tomht.Q, tomht.R)
print("Scan list:")
print(*scanList, sep = "\n", end = "\n\n")


for initialTarget in initialTargets:
 	tomht.initiateTrack(initialTarget)

associationHistory = []
for measurementIndex, measurementList in enumerate(scanList):
	associationHistory.append(tomht.addMeasurementList(measurementList))

# tomht.printTargetList()

plt.axis("equal")
plt.xlim((centerPosition.x-radarRange*1.05, centerPosition.x + radarRange*1.05))
plt.ylim((centerPosition.y-radarRange*1.05, centerPosition.y + radarRange*1.05))
# plt.show(block = True)
print("Done :)")