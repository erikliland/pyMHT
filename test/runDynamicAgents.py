#!/usr/bin/env python3
import os, sys
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['GUROBI_HOME'],'lib')
	#	export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
	
try:
	import signal 
	import time
	import functools
	import pulp
	import argparse
	import tomht
	import ast
	import logging
	from collections import namedtuple
	import tomht.radarSimulator as radarSimulator
	import tomht.helpFunctions as hpf
	import tomht.stateSpace.pv as model
	import xml.etree.ElementTree as ET
	import multiprocessing as mp 
	import simSettings as sim
	from compareResults import compareResults
	from plotTrackloss import plotTrackloss
	from plotRuntime import plotRuntime
except KeyboardInterrupt:
	print("Leaving already? Goodbye!")
	sys.exit()

simArgs = namedtuple('simArgs',['simList','loadLocation', 'fileString',
								'solver', 'lambda_phi', 'P_d', 'N', 
								'radarRange','p0','initialTargets'])
def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def runSimulation(sArgs,i):
	#simList, initialTargets, lambda_phi,lambda_nu,radarRange,p0,P_d,N,solver, i):
	try:
		seed = 5446 + i
		scanList = radarSimulator.simulateScans(
												seed, 
												sArgs.simList, 
												model.C, 
												model.R(model.sigmaR_true), 
												sArgs.lambda_phi,
												sArgs.radarRange, 
												sArgs.p0, 
												P_d = sArgs.P_d, 
												shuffle = True)
		tracker = tomht.Tracker(
								model.Phi, 
								model.C, 
								model.Gamma, 
								sArgs.P_d, 
								model.P0, 
								model.R(), 
								model.Q, 
								sArgs.lambda_phi, 
								sim.lambda_nu, 
								sim.eta2,
								sArgs.N, 
								sArgs.solver, 
								logTime = True,
								realTime = True
								)
		for initialTarget in sArgs.initialTargets:
		 	tracker.initiateTarget(initialTarget)

		covConsistenceList = []
		tic = time.time()
		for scanIndex, measurementList in enumerate(scanList):
			covConsistenceList.append( 
				tracker.addMeasurementList(measurementList, trueState = sArgs.simList[scanIndex]))
		toc = time.time()-tic
		trackList = hpf.backtrackNodePositions(tracker.__trackNodes__)
		if any ( len(track)-1 != len(sArgs.simList) for track in trackList):
			print(",",end = "", flush = True)
		else: 
			print(".",end = "", flush = True)
		return {'i':i, 
				'seed':seed, 
				'trackList':trackList, 
				'time':toc, 
				'runetimeLog': tracker.getRuntimeAverage(), 
				'covConsistence': covConsistenceList
				}
	except pulp.solvers.PulpSolverError as e:
		print("/",end = "", flush = True) 
	#except ValueError as e:
	#	print("v",end = "", flush = True)
	except OSError as e:
		print("O",end = "", flush = True)
	except KeyboardInterrupt:
		raise
	except Exception as e:
		print("tracker had an Exeption" + str(e))
		print("?",end = "", flush = True)
		raise


def runFile(root,iIter,sArgs,**kwargs):
	timeout = kwargs.get("t",60*5)
	runStart = time.time()
	simLog = 0.0
	nCores = min(max(1, kwargs.get("c", os.cpu_count() -1 )),os.cpu_count())
	pool = mp.Pool(nCores,initWorker)
	results = pool.imap_unordered(functools.partial(runSimulation,sArgs),iIter,1)
	while True:
		try:
			if time.time()-runStart > timeout*len(iIter):
				raise mp.TimeoutError 
			res = results.next(timeout=timeout)
			if res is not None:
				simLog += res['time']
				ET.SubElement(
					root, "Simulation", 
					i 				= repr(res.get('i')),
					seed 			= repr(res.get('seed')),
					totalSimTime 	= repr(res.get('time')), 
					runtimeLog 		= res.get('runetimeLog'), 
					covConsistence 	= [['{:.3e}'.format(v) for v in row] for row in res.get('covConsistence',[[]])],
					).text 			= repr(res.get('trackList'))
		except mp.TimeoutError:
			pool.close()
			pool.terminate()
			print(" Timed out after waiting", round(time.time()-runStart,1), "seconds")
			pool.join()
			time.sleep(1)
			break
		except StopIteration:
			pool.terminate()
			pool.join()
			break
	return (simLog, time.time()-runStart)

def simulateFile(sArgs,**kwargs):
	timeout = kwargs.get("t",60*5)
	try:
		savefilePath = (
			os.path.join(sArgs.loadLocation,
						os.path.splitext(sArgs.fileString)[0],
						"results",
						os.path.splitext(sArgs.fileString)[0])
						+"["
						+sArgs.solver.upper()
						+",Pd="+str(sArgs.P_d)
						+",N="+str(sArgs.N)
						+",lPhi="+'{:7.5f}'.format(sArgs.lambda_phi)
						+"]"
						+".xml")
		printFile = (	'{:51s}'.format(os.path.splitext(sArgs.fileString)[0])
						+", "+'{:6s}'.format(sArgs.solver)
						+", P_d="+str(sArgs.P_d)
						+", N="+str(sArgs.N)
						+", lPhi="+'{:5.0e}'.format(sArgs.lambda_phi)
						)
		nMonteCarlo = kwargs.get("i",sim.nMonteCarlo)
		if not os.path.exists(os.path.dirname(savefilePath)):
			os.makedirs(os.path.dirname(savefilePath))
		if (not os.path.isfile(savefilePath)) or kwargs.get("F",False):
			root = ET.Element("Simulations",
								lambda_nu = '{:.3e}'.format(sim.lambda_nu), 
								eta2 = '{:.2f}'.format(sim.eta2), 
								radarRange = '{:.3e}'.format(sArgs.radarRange), 
								p0 = repr(sArgs.p0),
								nMonteCarlo = str(nMonteCarlo),
								initialTargets = repr(sArgs.initialTargets)
								)
			print("S: ",printFile," ", sep = "", end = "", flush = True)
			(simTime, runTime) = runFile(root, range(nMonteCarlo), sArgs, **kwargs)
			print('@{0:5.0f} sec ({1:3.0f} sec)'.format(runTime, simTime))
			root.attrib["wallRunTime"] = repr(runTime)
			root.attrib["totalSimTime"]= repr(simTime)
			tree = ET.ElementTree(root)
			if not kwargs.get("D",False):
				tree.write(savefilePath)
			
		else:
			try:
				root 		= ET.parse(savefilePath).getroot()
			except ET.ParseError:
				os.remove(savefilePath)
				simulateFile(sArgs, **kwargs)
				return

			iList 		= [int(sim.get("i")) for sim in root.findall("Simulation")]
			iList.sort()
			preliminaryWallRunTime = float(root.attrib.get("wallRunTime",0))
			preliminaryTotalSimTime = float(root.attrib.get("totalSimTime",0))
			missingSimulationIndecies = set(range(nMonteCarlo)).difference(set(iList))
			if missingSimulationIndecies:
				print("P: ", printFile," ","V"*(len(iList)//5),"."*(len(iList)%5), sep = "", end = "", flush = True)
				(simTime, runTime) = runFile(root, missingSimulationIndecies, sArgs, **kwargs)
				print('@{0:5.0f} sec ({1:3.0f} sec) [{2:3.0f},{3:3.0f}] sec'.format(preliminaryWallRunTime+runTime, preliminaryTotalSimTime+simTime,preliminaryWallRunTime,preliminaryTotalSimTime))
				
				root.attrib["wallRunTime"] = repr(preliminaryWallRunTime + runTime)
				root.attrib["totalSimTime"]= repr(preliminaryTotalSimTime+ simTime)
				root.attrib["nMonteCarlo"] = repr(nMonteCarlo)
				if not kwargs.get("D",False):
					tree = ET.ElementTree(root)
					tree.write(savefilePath)
			else:
				computeTList= [float(sim.get("totalSimTime")) 	for sim in root.findall("Simulation")]
				runTime 	= root.attrib.get("wallRunTime")
				runTimeString = '{:5.0f}'.format(float(runTime)) if runTime is not None else "    ?"
				statusStringList = ['.' if i in iList else 'x' for i in range(len(iList))]
				statusString = "".join(statusStringList)
				statusString = statusString.replace(".....","V")
				timeStatus = '@'+runTimeString+' sec ({:3.0f} sec)'.format(sum(computeTList))
				print("J: ",printFile, ' {:40s}'.format(statusString),timeStatus,sep = "", flush = True)
	except KeyboardInterrupt:
		print("Killed by keyboard")
		raise
	except Exception as e :
		logging.error("simulateFile had en exception" + str(e))
		raise
		print("Failed" + e)

def runDynamicAgents(**kwargs):
	fileIndex = kwargs.get("f")
	if fileIndex is not None:
		files = [sim.simFiles[i] for i in fileIndex]
	else:
		files 	= sim.simFiles
	solvers = kwargs.get("s", sim.solvers)
	PdList 	= kwargs.get("p", sim.PdList)
	NList 	= kwargs.get("n", sim.NList)
	lambdaPhiList = kwargs.get("l", sim.lambdaPhiList)

	for solver in solvers:
		print('{:6s}'.format(solver),hpf.solverIsAvailable(solver))
	for fileString in files:
		filePath = os.path.join(sim.loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = radarSimulator.importFromFile(filePath)
		(p0, radarRange) = radarSimulator.findCenterPositionAndRange(simList)
		for solver in solvers:
			if not hpf.solverIsAvailable(solver):
				continue
			for P_d in PdList:
				for N in NList:
					for lambda_phi in lambdaPhiList:
						args = simArgs(	simList, 
										sim.loadLocation, 
										fileString, 
										solver, 
										lambda_phi, 
										P_d, 
										N, 
										radarRange,
										p0,
										initialTargets
										)
						# if not((N == 9) and (P_d != 0.5)):
						simulateFile(args,**kwargs)

if __name__ == '__main__':
	try:
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		parser = argparse.ArgumentParser(description = "Run MHT tracker simulations", argument_default=argparse.SUPPRESS)
		parser.add_argument('-F', help = "Force run of files (if exist)",action = 'store_true')
		parser.add_argument('-D', help = "Discard result", 				action = 'store_true') 
		parser.add_argument('-f', help = "File number to simulate", 	nargs = '+', type = int )
		parser.add_argument('-i', help = "Number of simulations", 		type = int )
		parser.add_argument('-c', help = "Number of cores to use",		type = int )
		parser.add_argument('-n', help = "Number of steps to remember",	nargs = '+', type = int )
		parser.add_argument('-s', help = "Solver for ILP problem",		nargs = '+')
		parser.add_argument('-p', help = "Probability of detection", 	nargs = '+', type = float)
		parser.add_argument('-l', help = "Lambda_Phi (noise)", 			nargs = '+', type = float)
		parser.add_argument('-t', help = "File simulation timeout",		type = float)
		parser.add_argument('-A', help = "Accumulate mode: continous loop of all files with increment of nCores", action = 'store_true')
		parser.add_argument('-b', help = "Batch size for accumulate mode in x*nCores, default = 1", type = int)
		parser.add_argument('-C', help = "Run compare and plot after finish", action = 'store_true')
		parser.add_argument('-m', help = "Start 'i' for Accumulate mode", type = int)
		args = vars(parser.parse_args())
		tomht._setHightPriority()
		tic = time.time()
		nCores = min(max(1, args.get("c", os.cpu_count() -1 )),os.cpu_count())
		print("Using", nCores, "workers")
		if args.get("A", False):
			iMax = args.get("i",float('inf'))
			iStep = nCores * args.get("b",1)
			iCurrent = args.get("m", 0)
			while iCurrent < iMax:
				iCurrent += iStep
				iCurrent = min(iCurrent, iMax)
				runDynamicAgents(**dict(args, i = iCurrent))
		else:
			runDynamicAgents(**args)
	except KeyboardInterrupt:
		print("Terminated", time.ctime())
		sys.exit()
	else:
		print("Finished(",round(time.time()-tic,1),"sec )@", time.ctime())
		if args.get("C",False):
			print("Runnig compare")
			compareResults()
			print("Plotting Trackloss")
			plotTrackloss()
			print("plotting Runtime")
			plotRuntime()
			print("Done!")
		sys.exit()
