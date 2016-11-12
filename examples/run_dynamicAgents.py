<<<<<<< HEAD
import os, sys, signal, time, getopt
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
from tomht.stateSpace.pv import *
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
	nMonteCarlo = 12
	nMonteCarlo = kwargs.get("nMonteCarlo", nMonteCarlo)
	lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
	sigma 		= 3						#Need to be changed to conficence

	for fileString in files:
		filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = sim.importFromFile(filePath)
		(p0, radarRange) = sim.findCenterPositionAndRange(simList)
		for solver in solvers:
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
										+solver
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
							results = pool.map(functools.partial(runSimulation,simList,initialTargets,C,R,Gamma,P0, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, Phi),range(nMonteCarlo))
							for res in results:
								simLog += res['time']
								ET.SubElement( root, "Simulation", i = str(res['i']),seed = str(res['seed']), totalSimTime = '{:.3e}'.format(res['time']), runtimeLog =res['runetimeLog'] ).text = repr(res['trackList'])
							print('@{0:5.1f}sec ({1:.1f} sec)'.format(time.clock()-runStart, simLog))
							tree = ET.ElementTree(root)
							tree.write(savefilePath)
						else:
							print("Jumped:     ",printFile, flush = True)

def runSimulation(simList,initialTargets,C,R,Gamma,P0, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, Phi, i):
	seed = 5446 + i
	scanList = sim.simulateScans(seed, simList, C, R, True, lambda_phi,radarRange, p0, P_d)
	tracker = tomht.Tracker(Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, lambda_nu, sigma, N, solver, logTime = True)
	for initialTarget in initialTargets:
	 	tracker.initiateTarget(initialTarget)
	tic = time.clock()
	for measurementList in scanList:
		tracker.addMeasurementList(measurementList, multiThread = False)
	toc = time.clock()-tic
	trackList = hpf.backtrackNodePositions(tracker.__trackNodes__)
	print(".",end = "", flush = True)
	return {'i':i, 'seed':seed, 'trackList':trackList, 'time':toc, 'runetimeLog':tracker.getRuntimeAverage()}

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
=======
import os, sys, signal, time, getopt
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
from tomht.stateSpace.pv import *
import xml.etree.ElementTree as ET
import multiprocessing as mp 
import functools

def signal_handler(signum, frame):
	time.sleep(1)
	print("You pressed Ctrl+C!")
	time.sleep(1)
	print("Goodbye")
	sys.exit(0)
#signal.signal(signal.SIGINT,signal_handler)

def runDynamicAgents(**kwargs):
	loadLocation = os.path.join("..","data")
	files = [	'dynamic_agents_full_cooperation.txt',
				'dynamic_agents_partial_cooporation.txt',
				'dynamic_and_static_agents_large_space.txt',
				'dynamic_and_static_agents_narrow_space.txt'
			]
	PdList = [0.5, 0.7, 0.9]
	NList = [1, 3, 6]
	lambdaPhiList = [0, 5e-5, 2e-4, 4e-4]
	solvers = ["CPLEX"]#,"GLPK","CBC","GUROBI"]
	nMonteCarlo = 10
	nMonteCarlo = kwargs.get("nMonteCarlo", nMonteCarlo)
	lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
	sigma 		= 3						#Need to be changed to conficence

	for fileString in files:
		filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = sim.importFromFile(filePath)
		(p0, radarRange) = sim.findCenterPositionAndRange(simList)
		for solver in solvers:
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
										+solver
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
							try:
								with mp.Pool() as pool:
									results = pool.map(functools.partial(runSimulation,simList,initialTargets,C,R,Gamma,P0, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, Phi),range(nMonteCarlo))
									for res in results:
										simLog += res['time']
										ET.SubElement( root, "Simulation", i = str(res['i']),seed = str(res['seed']), totalSimTime = '{:.3e}'.format(res['time']), runtimeLog =res['runetimeLog'] ).text = repr(res['trackList'])
								print('@{0:5.1f}sec ({1:.1f} sec)'.format(time.clock()-runStart, simLog))
								tree = ET.ElementTree(root)
								tree.write(savefilePath)
							except KeyboardInterrupt:
								print("Terminating Pool")
								pool.terminate()
								pool.join()
						else:
							print("Jumped:     ",printFile, flush = True)
				break
			break

def runSimulation(simList,initialTargets,C,R,Gamma,P0, lambda_phi,lambda_nu,radarRange,p0,P_d,sigma,N,solver, Phi, i):
	seed = 5446 + i
	scanList = sim.simulateScans(seed, simList, C, R, True, lambda_phi,radarRange, p0, P_d)
	tracker = tomht.Tracker(Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, lambda_nu, sigma, N, solver, logTime = True)
	for initialTarget in initialTargets:
	 	tracker.initiateTarget(initialTarget)
	tic = time.clock()
	for measurementList in scanList:
		tracker.addMeasurementList(measurementList, multiThread = False)
	toc = time.clock()-tic
	trackList = hpf.backtrackNodePositions(tracker.__trackNodes__)
	print(".",end = "", flush = True)
	return {'i':i, 'seed':seed, 'trackList':trackList, 'time':toc, 'runetimeLog':tracker.getRuntimeAverage()}

if __name__ == '__main__':
	argv = sys.argv[1:]
	nIterations = None
	try:
		opts, args = getopt.getopt(argv,"i:",["n="])
	except getopt.GetoptError:
		pass
	for opt, arg in opts:
		if opt == "-i":
			nIterations = int(arg)
	try:
		if nIterations is not None:
			runDynamicAgents(nMonteCarlo = nIterations)
		else:
			runDynamicAgents()
	except KeyboardInterrupt:
		print("Somebody killed me!!!")
		time.sleep(0.5)
>>>>>>> bdc854266624b31dcf021d5ed28f1a279df23d82
