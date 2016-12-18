#!/usr/bin/env python3
import os
import sys
import signal 
import time
import functools
import pulp
import csv
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

def getAverageRunTimeLog(fileString,solver,P_d, N, lambda_phi,**kwargs):
	filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
	(initialTargets, simList) = sim.importFromFile(filePath)
	(p0, radarRange) = sim.findCenterPositionAndRange(simList)
	printFile = (	'{:43s}'.format(os.path.splitext(fileString)[0])
					+'{:6s}'.format(solver)
					+", P_d="+str(P_d)
					+", N="+str(N)
					+", lPhi="+'{:5.0e}'.format(lambda_phi)
					)

	seed = 5446 + kwargs.get("i",0)
	scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = True)
	
	timeLog = []
	leafNodeTimeListLog = []
	for i in range(kwargs.get("j",10)):
		with tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R(), model.Q, lambda_phi, lambda_nu, eta2, N, solver,
							 realTime = True,logTime = True,debug = False, **kwargs) as tracker:
			for initialTarget in initialTargets:
			 	tracker.initiateTarget(initialTarget)

			for scanIndex, measurementList in enumerate(scanList):
				tracker.addMeasurementList(measurementList) #trueState = simList[scanIndex]
				if scanIndex == kwargs.get("k",1e15):
					break
			timeLog.append(tracker.toc)
			leafNodeTimeListLog.append(tracker.leafNodeTimeList)
		time.sleep(0.5)
	timeLogAvg = np.mean(np.array(timeLog,ndmin = 2), axis = 0)
	leafNodeTimeListLogAvg1 = np.mean(np.array(leafNodeTimeListLog, ndmin = 3), axis = 0)
	leafNodeTimeListLogAvg2= np.mean(leafNodeTimeListLogAvg1,axis = 0)
	#print(*leafNodeTimeListLog, sep = "\n")
	#print(leafNodeTimeListLogAvg1, "Avg1")
	#print(leafNodeTimeListLogAvg2, "Avg2")
	return timeLogAvg, leafNodeTimeListLogAvg2

if __name__ == '__main__':
	

	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	parser = argparse.ArgumentParser(description = "Run MHT tracker simulation with logging of runtime", argument_default=argparse.SUPPRESS)
	parser.add_argument('f', help = "File number to solve", type = int)
	parser.add_argument('s', help = "Solver for ILP problem")
	parser.add_argument('p', help = "Probability of detection",type = float)
	parser.add_argument('n', help = "Number of steps to keep history", type = int)
	parser.add_argument('l', help = "Lambda_Phi value (noise)", type = float)
	parser.add_argument('-i',help = "Random iteration selector", type = int) 
	parser.add_argument('-k',help = "Measurement number to stop at", type = int )
	parser.add_argument('-j',help = "Number of run to average", type = int)
	args = vars(parser.parse_args())
	print(args)

	np.set_printoptions(precision = 2, linewidth = 150, suppress = False)
	timeLogAvgList = []
	leafNodeTimeListLogAvgList = []
	for nProcesses in range(1,os.cpu_count()+1):
		print(nProcesses,"processes")
		timeLogAvg, leafNodeTimeListLogAvg = getAverageRunTimeLog(simFiles[args.get('f')],args.get('s'),args.get('p'),args.get('n'),args.get('l'),w = nProcesses, **args)
		#print(timeLogAvg,"Avg")
		timeLogAvgList.append( (nProcesses, timeLogAvg.tolist()) )
		tempList = [nProcesses, np.sum(leafNodeTimeListLogAvg)]
		tempList.extend(leafNodeTimeListLogAvg.tolist())
		leafNodeTimeListLogAvgList.append(tempList)

		print("-"*10)
		np.set_printoptions(precision = 2, linewidth = 150, suppress = True)
		with open("parallelTimeLog.csv",'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["Processes", "Total","Grow", "Cluster", "Optimize", "Prune"])
			for coreCountLog in timeLogAvgList:
				row = [coreCountLog[0]]
				row.extend(['{:.0f}'.format(elem*1000) for elem in coreCountLog[1]])
				print(row)
				writer.writerow(row)
		
		with open("parallelTimeLogPercentage.csv",'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["Processes","Grow", "Cluster", "Optimize", "Prune"])
			for coreCountLog in timeLogAvgList:
				totalTime = coreCountLog[1][0]
				row = [coreCountLog[0]]
				row.extend(['{:.0%}'.format(elem/totalTime) for elem in coreCountLog[1][1:]])
				print(row)
				writer.writerow(row)

		with open("parallelTimeLogDistribution.csv",'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["Processes","Search", "Predict", "Create", "Add"])
			for coreCountLog in leafNodeTimeListLogAvgList:
				if coreCountLog[0] == 1:
					continue
				totalTime = coreCountLog[1]
				row = [coreCountLog[0]]
				row.extend(['{:.0}'.format(elem/totalTime) for elem in coreCountLog[2:]])
				print(row)
				writer.writerow(row)