import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
from tomht.stateSpace.pv import *
import pulp

# file = './data/dynamic_agents_full_cooperation.txt'
readfile = '../data/dynamic_agents_partial_cooporation.txt'
(initialTargets, simList) = sim.importFromFile(readfile, startLine = 140*4)

seed = 5446
lambda_phi 	= 2e-4					#Expected number of false measurements per unit 
									# volume of the measurement space per scan
lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
									# of the measurement space per scan
P_d 		= 0.8					#Probability of detection
sigma 		= 3						#Need to be changed to conficence
N 		 	= 5						#Number of  timesteps to tail (N-scan)
solver  	= pulp.CPLEX_CMD(None, 0,1,0,[],0.05)
(p0, radarRange) = sim.findCenterPositionAndRange(simList)
scanList = sim.simulateScans(seed, simList, C, R, False, lambda_phi,radarRange, p0, P_d)


tracker = tomht.Tracker(Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, lambda_nu, sigma, N, solver)

for initialTarget in initialTargets:
 	tracker.initiateTarget(initialTarget)
for scanIndex, measurementList in enumerate(scanList):
	trackNodes = tracker.addMeasurementList(measurementList)
	if scanIndex == 50:
		break
savefile = os.path.splitext(readfile)[0] +"[Pd="+str(P_d)+",N="+str(N)+"]"+".txt"
trackList = hpf.backtrackNodePositions(trackNodes)
hpf.writeTracksToFile(savefile, trackList)