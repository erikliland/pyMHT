import os
import sys
import signal 
import time
import functools
import pulp

import xml.etree.ElementTree as ET
import multiprocessing as mp 
import matplotlib.pyplot as plt

import tomht
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
import tomht.stateSpace.pv as model
from simSettings import *

def runDynamicAgent(fileString,solver,P_d, N, lambda_phi,**kwargs):
	filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
	(initialTargets, simList) = sim.importFromFile(filePath)
	(p0, radarRange) = sim.findCenterPositionAndRange(simList)
	printFile = (	'{:43s}'.format(os.path.splitext(fileString)[0])
					+'{:6s}'.format(solver)
					+", P_d="+str(P_d)
					+", N="+str(N)
					+", lPhi="+'{:5.0e}'.format(lambda_phi)
					)

	print("Simulating: ",printFile, end = "", flush = True)
	runStart = time.time()
	simLog = 0.0
	
	seed = 5446
	scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = False)
	tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R(), model.Q, lambda_phi, lambda_nu, eta2, N, 1, solver, logTime = True)
	for initialTarget in initialTargets:
	 	tracker.initiateTarget(initialTarget)

	
	for scanIndex, measurementList in enumerate(scanList):
		tracker.addMeasurementList(measurementList, trueState = simList[scanIndex])
		if scanIndex == 50:
			break

	trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = True)
	
	association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
	#print("Association",*association, sep = "\n")

	fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
	hpf.plotRadarOutline(p0, radarRange, center = False)
	hpf.plotInitialTargets(initialTargets)
	hpf.plotVelocityArrowFromNode(tracker.__trackNodes__)
	hpf.plotValidationRegionFromNodes(tracker.__trackNodes__,tracker.eta2, 1)
	# hpf.plotValidationRegionFromForest(tracker.__targetList__, sigma, 1)
	# hpf.plotMeasurementsFromForest(tracker.__targetList__, real = True, dummy = True)
	hpf.plotMeasurementsFromList(tracker.__scanHistory__[-2:-1])
	hpf.plotMeasurementsFromNodes(tracker.__trackNodes__, labels = False, dummy = True)
	# hpf.plotHypothesesTrack(tracker.__targetList__)
	hpf.plotActiveTrack(tracker.__trackNodes__)
	hpf.plotTrueTrack(simList, markers = True)
	plt.axis("equal")
	plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
	plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
	plt.show()

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	runDynamicAgent(croppedFiles[0], solvers[0], 0.7,3,lambdaPhiList[2])
