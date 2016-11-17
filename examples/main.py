import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import pulp
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
from tomht.stateSpace.pv import*

def runSimulation():
	seed = 5446
	nTargets = 4
	p0 = Position(1,1)
	radarRange = 160.0 #meters
	meanSpeed = 10*scipy.constants.knot #meters/second
	initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, meanSpeed)
	initialTargets[3].state[2] 		*= -1
	initialTargets[3].state[2:4] 	*= 0.4
	initialTargets[1].state[2:4] 	*= 3
	# initialTargets[1].state[2] 		*= -1.5
	# initialTargets[1].state[2:4] 	*= 1.05
	print("Initial targets:")
	print(*initialTargets, sep='\n', end = "\n\n")

	nScans = 12
	timeStep = 1.0
	lambda_phi 	= 8e-4					#Expected number of false measurements per unit 
										# volume of the measurement space per scan
	lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
										# of the measurement space per scan
	P_d 		= 0.8					#Probability of detection
	confidence 	= 0.90					#
	N 		 	= 5						#Number of  timesteps to tail (N-scan)
	qSim 		= 0.8*q
	rSim 		= 0.8*r
	simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, Phi(timeStep), Q(timeStep,qSim), Gamma)

	print("Sim list:")
	print(*simList, sep = "\n", end = "\n\n")

	scanList = sim.simulateScans(seed, simList, C, np.eye(2)*rSim, lambda_phi,radarRange, p0, P_d = 1, shuffle = False)
	#solvers: CPLEX, GLPK, CBC, GUROBI
	tracker = tomht.Tracker(Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, lambda_nu, confidence, N, "CBC", logTime = True)

	# print("Scan list:")
	# print(*scanList, sep = "\n", end = "\n\n")

	for index, initialTarget in enumerate(initialTargets):
	 	tracker.initiateTarget(initialTarget)
	 	hpf.plotInitialTarget(initialTarget,index)

	for scanIndex, measurementList in enumerate(scanList):
		# print("#"*150)
		tracker.addMeasurementList(measurementList)
	print("#"*150)


	# hpf.printTargetList(tracker.__targetList__)
	association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
	print("Association",*association, sep = "\n")

	fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
	hpf.plotRadarOutline(p0, radarRange, center = False)
	hpt.plotInitialTargets(initialTargets)
	# hpf.plotVelocityArrowFromNode(tracker.__trackNodes__,2)
	hpf.plotValidationRegionFromNodes(tracker.__trackNodes__,tracker.eta2, 1)
	# hpf.plotValidationRegionFromForest(tracker.__targetList__, sigma, 1)
	# hpf.plotMeasurementsFromForest(tracker.__targetList__, real = True, dummy = True)
	hpf.plotMeasurementsFromList(tracker.__scanHistory__)
	# hpf.plotMeasurementsFromNodes(tracker.__trackNodes__)
	hpf.plotHypothesesTrack(tracker.__targetList__)
	hpf.plotActiveTrack(tracker.__trackNodes__)
	plt.axis("equal")
	plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
	plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
	plt.show()

if __name__ == '__main__':
	runSimulation()
	print("Done :)")
