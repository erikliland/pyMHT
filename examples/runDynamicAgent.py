#!/usr/bin/env python3
import os
import sys
import signal 
import time
import functools
import pulp
import argparse

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

	print( "Simulating: ", printFile, sep = " ", flush = True)
	runStart = time.time()
	simLog = 0.0
	
	seed = 5446 + kwargs.get("i",0)
	scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = False)
	tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R(), model.Q, lambda_phi, lambda_nu, eta2, 1, N, solver, logTime = True)
	for initialTarget in initialTargets:
	 	tracker.initiateTarget(initialTarget)

	if "t" in kwargs:
		fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
		plt.axis("equal")
		plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
		plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
		plt.show(block = False)
		timestep = kwargs.get("t")

	for scanIndex, measurementList in enumerate(scanList):
		tracker.addMeasurementList(measurementList, trueState = simList[scanIndex])

		if "t" in kwargs:
			plt.gca().cla()
			hpf.plotRadarOutline(p0, radarRange, center = False)
			tracker.plotInitialTargets()
			#tracker.plotValidationRegionFromRoot()
			tracker.plotValidationRegionFromTracks()
			tracker.plotScan()
			tracker.plotMeasurementsFromRoot(real = True, dummy = True,labels = False)
			tracker.plotHypothesesTrack()
			tracker.plotActiveTracks()
			hpf.plotTrueTrack(simList, markers = True)
			plt.axis("equal")
			plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
			plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
			fig1.canvas.draw()
			time.sleep(kwargs["t"])

	trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = True)
	association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
	#print("Association",*association, sep = "\n")
	
	plt.close()
	fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)	
#	hpf.plotRadarOutline(p0, radarRange, center = False)
	hpf.plotTrueTrack(simList, markers = True)
	tracker.plotInitialTargets()
#	tracker.plotVelocityArrowForTrack()
	# tracker.plotValidationRegionFromRoot()
	# tracker.plotValidationRegionFromTracks()
	# hpf.plotMeasurementsFromForest(tracker.__targetList__, real = True, dummy = True)
	# hpf.plotMeasurementsFromList(tracker.__scanHistory__)
#	tracker.plotMeasurementsFromTracks(labels = False, dummy = True)
#	tracker.plotHypothesesTrack()
#	tracker.plotActiveTracks()
	plt.axis("equal")
	plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
	plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
	fig1.canvas.draw()
	plt.show(block = True)

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	parser = argparse.ArgumentParser(description = "Run MHT tracker simulation", argument_default=argparse.SUPPRESS)
	parser.add_argument('f', help = "File number to solve", type = int)
	parser.add_argument('s', help = "Solver for ILP problem")
	parser.add_argument('p', help = "Probability of detection",type = float)
	parser.add_argument('n', help = "Number of steps to keep history", type = int)
	parser.add_argument('l', help = "Lambda_Phi value (noise)", type = float)
	parser.add_argument('-i',help = "Random iteration selector", type = int) 
	parser.add_argument('-t',help = "Step through the simulation", type = float )
	args = vars(parser.parse_args())
	print(args)
	runDynamicAgent(croppedFiles[args.get('f')],args.get('s'),args.get('p'),args.get('n'),args.get('l'), **args)