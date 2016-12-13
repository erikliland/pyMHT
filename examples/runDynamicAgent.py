#!/usr/bin/env python3
import os
import sys
import signal 
import time
import functools
import pulp
import argparse
import numpy as np

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
	
	simLog = 0.0
	
	seed = 5446 + kwargs.get("i",0)
	scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = False)
	tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R(), model.Q, lambda_phi, lambda_nu, eta2, N, solver, realTime = True, **kwargs)
	for initialTarget in initialTargets:
	 	tracker.initiateTarget(initialTarget)

	if "t" in kwargs:
		plt.figure(num=1, figsize = (9,9), dpi=100)
		plt.axis("equal")
		plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
		plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
		plt.ion()
		timestep = kwargs.get("t")

	runStart = time.time()
	try:
		for scanIndex, measurementList in enumerate(scanList):
			tracker.addMeasurementList(measurementList, trueState = simList[scanIndex])
			if scanIndex == kwargs.get("k",1e15):
				break

			if "t" in kwargs:
				plt.cla()
				hpf.plotRadarOutline(p0, radarRange, center = False)
				tracker.plotInitialTargets()
				plt.show(block = False)
				tracker.plotValidationRegionFromTracks()
				tracker.plotScan()
				tracker.plotMeasurementsFromRoot(real = True, dummy = True,labels = False)
				tracker.plotHypothesesTrack()
				tracker.plotActiveTracks()
				hpf.plotTrueTrack(simList)
				plt.axis("equal")
				plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
				plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
				plt.pause(kwargs["t"])

	except ValueError as e:
		tracker.printTargetList()
		print(e)
		raise
	runEnd = time.time()
	if not "t" in kwargs:
		runTime = runEnd-runStart
		print("Run time:", round(runTime,1), "sec")
		

	trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = True)
	association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
	#print("Association",*association, sep = "\n")
	
	plt.clf()
	hpf.plotRadarOutline(p0, radarRange, center = False)
	hpf.plotTrueTrack(simList)
	tracker.plotInitialTargets()
	tracker.plotVelocityArrowForTrack()
	# tracker.plotValidationRegionFromRoot()
	# tracker.plotValidationRegionFromTracks()
	# tracker.plotMeasurementsFromRoot(dummy = True)
	tracker.plotMeasurementsFromTracks(labels = False, dummy = True)
	# tracker.plotHypothesesTrack()
	tracker.plotActiveTracks()
	plt.axis("equal")
	plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
	plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
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
	parser.add_argument('-k',help = "Measurement number to stop at", type = int )
	parser.add_argument('-S',help = "Run tracker in single thread", action = 'store_false')
	args = vars(parser.parse_args())
	print(args)
	runDynamicAgent(simFiles[args.get('f')],args.get('s'),args.get('p'),args.get('n'),args.get('l'), **args)