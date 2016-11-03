from classDefinitions import * 
import helpFunctions as hpf
import radarSimulator as sim
import tomht
import os

# file = './data/dynamic_agents_full_cooperation.txt'
readfile = './data/dynamic_agents_partial_cooporation.txt'
(initialTargets, simList) = sim.importFromFile(readfile, startLine = 140*4)

seed = 5446
(p0, radarRange) = sim.findCenterPositionAndRange(simList)
scanList = sim.simulateScans(seed, simList, tomht.C, tomht.R, False, 
									tomht.lambda_phi,radarRange, p0, tomht.P_d)

for initialTarget in initialTargets:
 	tomht.initiateTarget(initialTarget)
associationHistory = []
for scanIndex, measurementList in enumerate(scanList):
	associationHistory.append(tomht.addMeasurementList(measurementList))
savefile = os.path.splitext(readfile)[0] +"[Pd="+str(tomht.P_d)+",N="+str(tomht.N)+"]"+".txt"
trackList = hpf.backtrackNodePositions(associationHistory[-1])
hpf.writeTracksToFile(savefile, trackList)
