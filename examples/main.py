#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import pulp
import time
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
from tomht.stateSpace.pv import*

def runSimulation():
	seed = 5446+1
	nTargets = 4
	p0 = Position(1,1)
	radarRange = 160.0 #meters
	meanSpeed = 10*scipy.constants.knot #meters/second
	# initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, meanSpeed)
	# initialTargets[3].state[2] 		*= -0.5
	# initialTargets[3].state[2:4] 	*= 0.4
	# initialTargets[1].state[2:4] 	*= 3
	# initialTargets[1].state[2] 		*= 10
	initialTargets = []
	initialTargets.append(sim.SimTarget(np.array([-50,-100, 1,9]),time.time()))
	initialTargets.append(sim.SimTarget(np.array([-25,-100,-1,8]),time.time()))
	initialTargets.append(sim.SimTarget(np.array([  0,-100, 0,8]),time.time()))
	initialTargets.append(sim.SimTarget(np.array([  5,-100, 0,8]),time.time()))
	initialTargets.append(sim.SimTarget(np.array([ 10,-100, 0,8]),time.time()))
	
	# initialTargets[1].state[2:4] 	*= 1.05
	print("Initial targets:")
	print(*initialTargets, sep='\n', end = "\n\n")

	simTime = 30 #sec
	timeStep = 2.0
	nScans = round(simTime/timeStep)
	lambda_phi 	= 4e-4		#Expected number of false measurements per unit 
							# volume of the measurement space per scan
	lambda_nu 	= 0.0001	#Expected number of new targets per unit volume 
							# of the measurement space per scan
	P_d 		= 0.8		#Probability of detection
	N 		 	= 5			#Number of  timesteps to tail (N-scan)
	eta2 		= 5.99 		#95% confidence
	pruneThreshold = sigmaR_tracker

	simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, Phi(timeStep), Q(timeStep,sigmaQ_true), Gamma)
	# print("Sim list:")
	# print(*simList, sep = "\n", end = "\n\n")
	sim.writeSimList(initialTargets, simList, "parallel_targets_0.5Hz.txt")
	
	scanList = sim.simulateScans(seed, simList, C, R(sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = False)
	#solvers: CPLEX, GLPK, CBC, GUROBI
	tracker = tomht.Tracker(Phi, C, Gamma, P_d, P0, R(), Q, lambda_phi, lambda_nu, eta2, pruneThreshold, N, "CBC", logTime = True)

	# print("Scan list:")
	# print(*scanList, sep = "\n", end = "\n\n")

	for index, initialTarget in enumerate(initialTargets):
	 	tracker.initiateTarget(initialTarget)
	for scanIndex, measurementList in enumerate(scanList):
		break
		tracker.addMeasurementList(measurementList, printTime = True)
	print("#"*150)

	# hpf.printTargetList(tracker.__targetList__, backtrack = True)
	association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
	print("Association",*association, sep = "\n")

	fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
	# hpf.plotRadarOutline(p0, radarRange, center = False)
	hpf.plotTrueTrack(simList)
	tracker.plotInitialTargets()
	# tracker.plotVelocityArrowForTrack()
	# tracker.plotValidationRegionFromRoot()
	# tracker.plotValidationRegionFromTracks()
	# tracker.plotMeasurementsFromRoot(dummy = True)
	# tracker.plotMeasurementsFromTracks(labels = False, dummy = True)
	# tracker.plotHypothesesTrack()
	# tracker.plotActiveTracks()
	plt.axis("equal")
	plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
	plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
	fig1.canvas.draw()
	plt.show(block = True)

if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	runSimulation()
	print("Done :)")
