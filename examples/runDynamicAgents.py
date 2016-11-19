import os
import sys
import signal 
import time
import functools
import pulp
import argparse
import tomht
import tomht.radarSimulator as radarSimulator
import tomht.helpFunctions as hpf
import tomht.stateSpace.pv as model
import xml.etree.ElementTree as ET
import multiprocessing as mp 
import simSettings as sim

def runSimulation(simList, initialTargets, lambda_phi,lambda_nu,radarRange,
							p0,P_d,N,solver,timeoutDuration, i):
	try:
		seed = 5446 + i
		scanList = radarSimulator.simulateScans(
												seed, 
												simList, 
												model.C, 
												model.R(model.sigmaR_true), 
												lambda_phi,
												radarRange, 
												p0, 
												P_d = P_d, 
												shuffle = True)
		tracker = tomht.Tracker(
								model.Phi, 
								model.C, 
								model.Gamma, 
								P_d, 
								model.P0, 
								model.R(), 
								model.Q, 
								lambda_phi, 
								lambda_nu, 
								sim.eta2, 
								model.sigmaR_tracker, 
								N, 
								solver, 
								logTime = True
								)
		for initialTarget in initialTargets:
		 	tracker.initiateTarget(initialTarget)

		covConsistenceList = []
		tic = time.process_time()
		for scanIndex, measurementList in enumerate(scanList):
			covConsistenceList.append( 
				tracker.addMeasurementList(
					measurementList, trueState = simList[scanIndex], multiThread = False)
				)
		toc = time.process_time()-tic
		trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = False)
		if any ( len(track)-1 != len(simList) for track in trackList):
			print(",",end = "", flush = True)
		else: 
			print(".",end = "", flush = True)
		res =  {'i':i, 
				'seed':seed, 
				'trackList':trackList, 
				'time':toc, 
				'runetimeLog':tracker.getRuntimeAverage(), 
				'covConsistence': covConsistenceList
				}
	except pulp.solvers.PulpSolverError:
		print("/",end = "", flush = True)
		res = None 
	except ValueError:
		print("v",end = "", flush = True)
		res = None 
	except OSError:
		print("O",end = "", flush = True)
	except KeyboardInterrupt:
		raise
	except:
		print("?",end = "", flush = True)
	return res

def simulateFile(simList, loadLocation, fileString, solver, lambda_phi, P_d, N, 
										radarRange,p0,initialTargets, **kwargs):
	timeout = kwargs.get("timeout",100)
	try:
		savefilePath = (os.path.join(loadLocation,os.path.splitext(fileString)[0],"results",os.path.splitext(fileString)[0])
						+"["
						+solver.upper()
						+",Pd="+str(P_d)
						+",N="+str(N)
						+",lPhi="+'{:7.5f}'.format(lambda_phi)
						+"]"
						+".xml")
		printFile = (	'{:43s}'.format(os.path.splitext(fileString)[0])
						+'{:6s}'.format(solver)
						+", P_d="+str(P_d)
						+", N="+str(N)
						+", lPhi="+'{:5.0e}'.format(lambda_phi)
						)
		if (not os.path.isfile(savefilePath)) or kwargs.get("F",False):
			nMonteCarlo = kwargs.get("i",sim.nMonteCarlo)
			root = ET.Element("Simulations",
								lambda_nu = '{:.3e}'.format(sim.lambda_nu), 
								eta2 = '{:.2f}'.format(sim.eta2), 
								radarRange = '{:.3e}'.format(radarRange), 
								p0 = repr(p0),
								nMonteCarlo = str(nMonteCarlo),
								initialTargets = repr(initialTargets)
								)
			print("Simulating: ",printFile, end = "", flush = True)
			runStart = time.time()
			simLog = 0.0
			results = pool.map_async(
				functools.partial(runSimulation,simList,initialTargets,lambda_phi,sim.lambda_nu,radarRange,p0,P_d,N,solver,1),
				range(nMonteCarlo),
				1)
			results.wait(timeout = timeout)
			if results.ready():
				for res in results.get(timeout = 1):
					if res is not None:
						simLog += res['time']
						ET.SubElement( root, "Simulation", 
										i 				= repr(res.get('i')),
										seed 			= repr(res.get('seed')), 
										totalSimTime 	= '{:.3}'.format((res.get('time'))), 
										runtimeLog 		= res.get('runetimeLog'), 
										covConsistence 	= [['{:.3e}'.format(v) for v in row] for row in res.get('covConsistence',[[]])],
										).text 			= repr(res.get('trackList'))
				print('@{0:5.1f}sec ({1:.1f} sec)'.format(time.time()-runStart, simLog))
				tree = ET.ElementTree(root)
				if not kwargs.get("D",False):
					if not os.path.exists(os.path.dirname(savefilePath)):
						os.makedirs(os.path.dirname(savefilePath))
					tree.write(savefilePath)
			else:
				print("Timed out after",timeout, "seconds")
		else:
			print("Jumped:     ",printFile, flush = True)
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		raise
	except mp.TimeoutError:
		print("Timed out after", timeout, "seconds")
		pool.terminate()
		pool.join()
	except Exception as e :
		raise
		print("Failed", e)

def runDynamicAgents(pool, **kwargs):
	fileIndex = kwargs.get("f")
	if fileIndex is not None:
		files = [sim.croppedFiles[i] for i in fileIndex]
	else:
		files 	= sim.croppedFiles
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
						simulateFile(	simList, 
										sim.loadLocation, 
										fileString, 
										solver, 
										lambda_phi, 
										P_d, 
										N, 
										radarRange,
										p0,
										initialTargets,
										**kwargs)

def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
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
	args = vars(parser.parse_args())
	tic = time.time()
	try:
		nCores = args.get("c", os.cpu_count() -1 )
		print("Using", nCores, "workers")
		pool = mp.Pool(nCores,initWorker)
		runDynamicAgents(pool, **args)
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		print("Terminated", time.ctime())
	else:
		pool.close()
		pool.join()
		print("Finished(",round(time.time()-tic,1),"sec )@", time.ctime())
