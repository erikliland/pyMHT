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

def runSimulation(simList, initialTargets, lambda_phi,lambda_nu,radarRange,p0,P_d,N,solver, i):
	try:
		seed = 5446 + i
		scanList = radarSimulator.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true), lambda_phi,radarRange, p0, P_d = P_d, shuffle = True)
		tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R(), model.Q, lambda_phi, lambda_nu, sim.eta2, model.sigmaR_tracker, N, solver, logTime = True)
		for initialTarget in initialTargets:
		 	tracker.initiateTarget(initialTarget)

		covConsistenceList = []
		tic = time.clock()
		for scanIndex, measurementList in enumerate(scanList):
			covConsistenceList.append( tracker.addMeasurementList(measurementList, trueState = simList[scanIndex], multiThread = False) )
		toc = time.clock()-tic
		trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = True)
		if any ( len(track)-1 != len(simList) for track in trackList):
			print(",",end = "", flush = True)
		else: 
			print(".",end = "", flush = True)
		return {'i':i, 
				'seed':seed, 
				'trackList':trackList, 
				'time':toc, 
				'runetimeLog':tracker.getRuntimeAverage(), 
				'covConsistence': covConsistenceList
				}
	except pulp.solvers.PulpSolverError:
		print("/",end = "", flush = True)
		return 

def simulateFile(simList, loadLocation, fileString, solver, lambda_phi, P_d, N, radarRange,p0,initialTargets, **kwargs):
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
		results = pool.map(functools.partial(runSimulation,simList,initialTargets,lambda_phi,sim.lambda_nu,radarRange,p0,P_d,N,solver),range(nMonteCarlo))
		for res in results:
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
		if not os.path.exists(os.path.dirname(savefilePath)):
			os.makedirs(os.path.dirname(savefilePath))
		tree.write(savefilePath)
	else:
		print("Jumped:     ",printFile, flush = True)

def runDynamicAgents(pool, **kwargs):
	for solver in sim.solvers:
		print('{:6s}'.format(solver),hpf.solverIsAvailable(solver))
	for fileString in sim.files:
		filePath = os.path.join(sim.loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = radarSimulator.importFromFile(filePath)
		(p0, radarRange) = radarSimulator.findCenterPositionAndRange(simList)
		for solver in sim.solvers:
			if not hpf.solverIsAvailable(solver):
				print("Checking solver",solver,": Failed")
				continue
			for P_d in sim.PdList:
				for N in sim.NList:
					for lambda_phi in sim.lambdaPhiList:
						simulateFile(simList, sim.loadLocation, fileString, solver, lambda_phi, P_d, N, radarRange,p0,initialTargets, **kwargs)

def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	parser = argparse.ArgumentParser(description = "Run MHT tracker simulations", argument_default=argparse.SUPPRESS)
	parser.add_argument('-F', help = "Force overwrite of files", action = 'store_true')
	parser.add_argument('-i', help = "Number of simulations", type = int)
	parser.add_argument('-c', help = "Number of cores to use",type = int)
	args = vars(parser.parse_args())
	print(args)
	tic = time.clock()
	try:
		pool = mp.Pool(args.get("c",os.cpu_count()),initWorker)
		runDynamicAgents(pool, **args)
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		print("Terminated", time.ctime())
	else:
		pool.close()
		pool.join()
		print("Finished(",round(time.clock()-tic,1),"sec )@", time.ctime())
