import os, sys, signal, time, getopt
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
import xml.etree.ElementTree as ET
import multiprocessing as mp 
import functools
import pulp
from simSettings import *

def runDynamicAgents(pool, **kwargs):
	# nMonteCarlo = kwargs.get("nMonteCarlo", nMonteCarlo)
	for fileString in files:
		filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = sim.importFromFile(filePath)
		(p0, radarRange) = sim.findCenterPositionAndRange(simList)
		for solver in solvers:
			if not hpf.solverIsAvailable(solver):
				print("Checking solver",solver,": Failed")
				continue
			for P_d in PdList:
				for N in NList:
					for lambda_phi in lambdaPhiList:
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
						if not os.path.isfile(savefilePath):
							root = ET.Element("Simulations",
												lambda_nu = '{:.3e}'.format(lambda_nu), 
												sigma = '{:.3e}'.format(sigma), 
												radarRange = '{:.3e}'.format(radarRange), 
												p0 = repr(p0),
												nMonteCarlo = str(nMonteCarlo),
												initialTargets = repr(initialTargets)
												)
							print("Simulating: ",printFile, end = "", flush = True)
							runStart = time.time()
							simLog = 0.0
							results = pool.map(functools.partial(runSimulation,simList,initialTargets,lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver),range(nMonteCarlo))
							for res in results:
								if res is not None:
									simLog += res['time']
									ET.SubElement( root, "Simulation", 
													i 				= repr(res.get('i')),
													seed 			= repr(res.get('seed')), 
													totalSimTime 	= repr(res.get('time')), 
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

def runSimulation(simList,initialTargets, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, i):
	try:
		import tomht.stateSpace.pv as model
		seed = 5446 + i
		scanList = sim.simulateScans(seed, simList, model.C, model.R, True, lambda_phi,radarRange, p0, P_d)
		tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R, model.Q, lambda_phi, lambda_nu, sigma, N, solver, logTime = True)
		for initialTarget in initialTargets:
		 	tracker.initiateTarget(initialTarget)

		covConsistenceList = []
		tic = time.clock()
		for scanIndex, measurementList in enumerate(scanList):
			covConsistenceList.append( tracker.addMeasurementList(measurementList, trueState = simList[scanIndex], multiThread = False) )
		toc = time.clock()-tic
		trackList = hpf.backtrackNodePositions(tracker.__trackNodes__, debug = True)
		if any ( len(track) != 303 for track in trackList):
			print(",",end = "", flush = True)
		else: 
			print(".",end = "", flush = True)
		return {'i':i, 'seed':seed, 'trackList':trackList, 'time':toc, 'runetimeLog':tracker.getRuntimeAverage(), 'covConsistence': covConsistenceList}
	except pulp.solvers.PulpSolverError:
		print("/",end = "", flush = True)
		return 

def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
	import time
	tic = time.clock()
	try:
		argv = sys.argv[1:]
		nIterations = None
		try:
			opts, args = getopt.getopt(argv,"i:",["n="])
		except getopt.GetoptError:
			pass
		for opt, arg in opts:
			if opt == "-i":
				nIterations = int(arg)
		pool = mp.Pool(os.cpu_count(),initWorker)
		if nIterations is not None:
			runDynamicAgents(pool, nMonteCarlo = nIterations)
		else:
			runDynamicAgents(pool)
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		print("Terminated", time.ctime())
	else:
		pool.close()
		pool.join()
		print("Finished(",round(time.clock()-tic,1),"sec )@", time.ctime())
