from classDefinitions import *
import random
import matplotlib.pyplot as plt

#Create simulated targets
numOfTargets = 4
numOfScans = 5
measurmentMatrix = generateMeasurementMatrix(numOfTargets,numOfScans)

print("Number of targets in scan:",len(measurmentMatrix[0]))
for targetIndex in range(numOfTargets):
	xList = []
	yList = []
	for scanIndex in range(numOfScans):
		xList.append( measurmentMatrix[scanIndex][targetIndex].Position.x )
		yList.append( measurmentMatrix[scanIndex][targetIndex].Position.y )
	plt.plot(xList, yList,"->")

plt.show()