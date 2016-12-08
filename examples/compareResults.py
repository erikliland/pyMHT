#!/usr/bin/env python3
import os, sys
import matplotlib.pyplot as plt
import tomht.radarSimulator as sim
import numpy as np
import xml.etree.ElementTree as ET
from simSettings import *
import ast
import datetime
import signal
import functools
import multiprocessing as mp 


def openRes(filename):
	try:
		f = open(filename)
		trackList = []
		for track in f:
			posStringArray = "".join(track.split()).strip(" ( ) ").split("),(")
			trackArray = []
			for pos in posStringArray:
				trackArray.append( np.array( pos.split(","), dtype = float) )
			trackList.append(np.asarray(trackArray))
		return np.asarray(trackList)
	except:
		return

def openGroundTruth(filename):
	(initialTargets, simList) = sim.importFromFile(filename)
	nTargets = len(initialTargets)
	nTimestep= len(simList)
	targetTracks = np.zeros((nTargets,nTimestep+1,2))
	for target in range(nTargets):
		for timestep in range(nTimestep+1):
			if timestep == 0:
				targetTracks[target,timestep,:] = initialTargets[target].state[0:2]
			else:
				targetTracks[target,timestep,:] = simList[timestep-1][target].state[0:2]
	return np.asarray(targetTracks)

def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def analyzeSimulation(nTracksTrue, trueTrackLength, trueTracks,threshold, simulation):
	##Function start, input = (simulation, nTracksTrue, trueTrackLength, trueTracks,threshold)
	parsedTracks = ast.literal_eval(simulation.text)
	estimatedTracks = np.array(parsedTracks)
	if nTracksTrue != len(parsedTracks):
		return {'status':"/"}
	
	if any(len(track) != trueTrackLength for track in parsedTracks):
		return {'status':"/"}

	if estimatedTracks is None:
		return None

	if trueTracks.shape != estimatedTracks.shape:
		return {'status':"o"}
	lostTracks = np.linalg.norm(trueTracks-estimatedTracks,2,2) > threshold
	lostTracksTime = [np.flatnonzero(lostTrack).tolist() for lostTrack in lostTracks]
	runTimeLog = ast.literal_eval(simulation.get("runtimeLog"))
	return {'status':'.',
			'lostTracks':lostTracks,
			'lostTracksTime':lostTracksTime,
			'estimatedTracks':estimatedTracks,
			'runTimeLog':runTimeLog}
	##Function end, return status, lostTracks, lostTracksTime, runTimeLog

def compareResults():
	sumTotalSimTime = 0
	sumTotalWallRunTime = 0
	root = ET.Element("simulations")
	pool = mp.Pool(os.cpu_count(),initWorker)
	for fileString in simFiles:
		filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
		doc = ET.SubElement(root,"file", name = os.path.basename(fileString))
		trueTracks = openGroundTruth(filePath)
		trueTrackLength = len(trueTracks[0])
		nTracksTrue = trueTracks.shape[0]
		for solver in solvers:
			solv = ET.SubElement(doc, "Solver", name = solver)
			for P_d in PdList:
				prob = ET.SubElement(solv, "P_d", value = str(P_d))
				for N in NList:
					num = ET.SubElement(prob, "N", value = str(N))
					for lambda_phi in lambdaPhiList:
						nTracks = 0
						nLostTracks = 0
						runTimeLogAvg = {}
						print('{:45s}'.format(os.path.splitext(fileString)[0]),'{:6s}'.format(solver),"P_d =",P_d,"N =",N,"lPhi =",'{:5.0e}'.format(lambda_phi), end = "")
						savefilePath = (os.path.join(loadLocation,os.path.splitext(fileString)[0],"results",os.path.splitext(fileString)[0])
											+"["
											+solver.upper()
											+",Pd="+str(P_d)
											+",N="+str(N)
											+",lPhi="+'{:7.5f}'.format(lambda_phi)
											+"]"
											+".xml")
						try:
							simulations = ET.parse(savefilePath).getroot()
						except FileNotFoundError:
							print('{:80s}'.format("Not found"))
							continue
						except ET.ParseError:
							os.remove(savefilePath)
							print("Deleted corrupt file:", savefilePath)
							continue

						iList = [int(sim.get("i")) for sim in simulations.findall("Simulation")]
						totalSimTime = sum([float(sim.get("totalSimTime")) for sim in simulations.findall("Simulation")])
						sumTotalWallRunTime += float(simulations.attrib.get("wallRunTime"))
						sumTotalSimTime += totalSimTime
						iList.sort() 
						missingSimulationIndecies = set(range(nMonteCarlo)).difference(set(iList))
						nSimulations = int(simulations.attrib.get('nMonteCarlo'))
						statusString = ""
						results = pool.imap_unordered(functools.partial(analyzeSimulation, nTracksTrue, trueTrackLength, trueTracks,threshold),simulations,1)
						while True:
							try:
								res = results.next()
								if res is not None:
									status = res.get("status")
									if (status is not None) and (status == "."):
										runTimeLog = res.get("runTimeLog")
										lostTracksTime = res.get("lostTracksTime")
										estimatedTracks = res.get("estimatedTracks")
										for k,v in runTimeLog.items():
											try:
												runTimeLogAvg[k] += v
											except KeyError:
												runTimeLogAvg[k] = v
										permanentLostTracks = []

										for lostTrackTime in lostTracksTime:
											if len(lostTrackTime):
												permanentLostTracks.append(lostTrackTime[-1] == trueTrackLength-1)
										nLostTracks += sum(permanentLostTracks)
										nTracks += len(estimatedTracks)
										statusString += status
							except StopIteration:
								# pool.terminate()
								# pool.join()
								break
						# statusString = statusString.replace(".....","V")	
						# print('{:70s}'.format(statusString), end = "")

						if nTracks != 0:
							print("\t",'{:3.0f}'.format(nLostTracks),"/",'{:3.0f}'.format(nTracks),"=>",'{:4.1f}'.format((nLostTracks/nTracks)*100),"%")
							lambdaPhi = ET.SubElement(num,"lambda_phi", value = '{:5.0e}'.format(lambda_phi))
							ET.SubElement(lambdaPhi,"nTracks").text 	= repr(nTracks)
							ET.SubElement(lambdaPhi,"nLostTracks").text = repr(nLostTracks)
							ET.SubElement(lambdaPhi,"totalTime").text 	= repr(totalSimTime)
							ET.SubElement(lambdaPhi,"runTimeLogAvg").text = repr(runTimeLogAvg)
							ET.SubElement(lambdaPhi,"nSimulations").text= repr(nSimulations)
							# ET.SubElement(lambdaPhi,"covConsistence").text = simulation.get("covConsistence")
						else:
							print()
									
	root.attrib["sumTotalSImTime"] = repr(sumTotalSimTime)
	root.attrib["sumTotalWallRunTime"] = repr(sumTotalWallRunTime)
	tree = ET.ElementTree(root)
	tree.write("compareResult.xml")
	print("SumTotalSimTime",str(datetime.timedelta(seconds=sumTotalSimTime)))
	print("SumTotalWallRunTime",str(datetime.timedelta(seconds=sumTotalWallRunTime)))

if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	compareResults()