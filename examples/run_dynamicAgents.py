import os, sys, signal, time, getopt
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
import xml.etree.ElementTree as ET
import multiprocessing as mp 
import functools

def runDynamicAgents(pool, **kwargs):
	loadLocation = os.path.join("..","data")
	files = [	'dynamic_agents_full_cooperation.txt',
				'dynamic_agents_partial_cooporation.txt',
				'dynamic_and_static_agents_large_space.txt',
				'dynamic_and_static_agents_narrow_space.txt'
			]
	PdList = [0.5, 0.7, 0.9]
	NList = [1, 3, 6]
	lambdaPhiList = [0, 5e-5, 2e-4, 4e-4]
	solvers = ["CPLEX","GLPK","CBC","GUROBI"]
	nMonteCarlo = 11
	nMonteCarlo = kwargs.get("nMonteCarlo", nMonteCarlo)
	lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
	sigma 		= 3						#Need to be changed to conficence

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
							runStart = time.clock()
							simLog = 0.0
							results = pool.map(functools.partial(runSimulation,simList,initialTargets,lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver),range(nMonteCarlo))
							for res in results:
								if res is not None:
									simLog += res['time']
									ET.SubElement( root, "Simulation", i = str(res['i']),seed = str(res['seed']), totalSimTime = '{:.3e}'.format(res['time']), runtimeLog =res['runetimeLog'] ).text = repr(res['trackList'])
							print('@{0:5.1f}sec ({1:.1f} sec)'.format(time.clock()-runStart, simLog))
							tree = ET.ElementTree(root)
							if not os.path.exists(os.path.dirname(savefilePath)):
								os.makedirs(os.path.dirname(savefilePath))
							tree.write(savefilePath)
						else:
							print("Jumped:     ",printFile, flush = True)

def runSimulation(simList,initialTargets, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, i):
	import pulp
	try:
		import tomht.stateSpace.pv as model
		seed = 5446 + i
		scanList = sim.simulateScans(seed, simList, model.C, model.R, True, lambda_phi,radarRange, p0, P_d)
		tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, P_d, model.P0, model.R, model.Q, lambda_phi, lambda_nu, sigma, N, solver, logTime = True)
		for initialTarget in initialTargets:
		 	tracker.initiateTarget(initialTarget)
		tic = time.clock()
		for measurementList in scanList:
			tracker.addMeasurementList(measurementList, multiThread = False)
		toc = time.clock()-tic
		trackList = hpf.backtrackNodePositions(tracker.__trackNodes__)
		print(".",end = "", flush = True)
		return {'i':i, 'seed':seed, 'trackList':trackList, 'time':toc, 'runetimeLog':tracker.getRuntimeAverage()}
	except pulp.solvers.PulpSolverError:
		print("/",end = "", flush = True)
		return 

def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
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
		print("Terminated")
	else:
		pool.close()
		pool.join()
		print("Finish")
