import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pulp
import cProfile
import pstats
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import tomht
from tomht.classDefinitions import Position
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf

seed = 5446
nTargets = 4
p0 = Position(1,1)
radarRange = 10.0 #meters
maxSpeed = 2 #meters/second
initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, maxSpeed)
initialTargets[3].state[3] *= -1
initialTargets[3].state[2:4] *= 0.3
initialTargets[1].state[2:4] *= 1.5
initialTargets[1].state[2] *= -1.5
initialTargets[1].state[2:4] *= 1.05
print("Initial targets:")
print(*initialTargets, sep='\n', end = "\n\n")

nScans = 3
timeStep = 1.0
T 		= timeStep
b 		= np.zeros(4) 				#Transition offsets
C 		= np.array([[1, 0, 0, 0],	#Also known as "H"
					[0, 1, 0, 0]])	
d 		= np.zeros(2)				#Observation offsets
Gamma 	= np.diag([1,1],-2)[:,0:2]	#Disturbance matrix (only velocity)
P_d 	= 1.0						#Probability of detection
p 		= np.power(1e-2,1)			#Initial systen state variance
P0 		= np.diag([p,p,p,p])		#Initial state covariance
r		= np.power(1e-3,1)			#Measurement variance
q 		= np.power(2e-2,1)			#Velocity variance variance
R 		= np.eye(2) * r 			#Measurement/observation covariance
Q		= np.eye(2) * q * T 		#Transition/system covariance (process noise)
lambda_phi 	= 2e-4					#Expected number of false measurements per unit 
									# volume of the measurement space per scan
lambda_nu 	= 0.0001				#Expected number of new targets per unit volume 
									# of the measurement space per scan
lambda_ex 	= lambda_phi+lambda_nu 	#Spatial density of the extraneous measurements
sigma 		= 3						#Need to be changed to conficence
sigma2		= np.power(sigma,2)
N 		 	= 5						#Number of  timesteps to tail (N-scan)
problemSizeLimit = 10000
solver  	= pulp.CPLEX_CMD(None, 0,1,0,[],0.05)
# solver  	= pulp.GLPK_CMD(None, 0,1,0,[])
# solver  	= pulp.PULP_CBC_CMD()
# solver  	= pulp.SYMPHONY()	#Unavailable
# solver  	= pulp.GUROBI_CMD(None, 0,1,0,[])
# solver  	= pulp.XPRESS()		#Unavailable
def Phi(T):
	from numpy import array
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])


simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, Phi(timeStep), Q, Gamma)

print("Sim list:")
print(*simList, sep = "\n", end = "\n\n")

scanList = sim.simulateScans(seed, simList, C, R, False, lambda_phi,radarRange, p0)

tracker = tomht.Tracker(timeStep, Phi, C, Gamma, P_d, P0, R, Q, lambda_phi, lambda_nu, sigma, N, solver)

# print("Scan list:")
# print(*scanList, sep = "\n", end = "\n\n")

for initialTarget in initialTargets[0:3]:
 	tracker.initiateTarget(initialTarget)

for scanIndex, measurementList in enumerate(scanList):
	print("#"*150)
	trackNodes = tracker.addMeasurementList(measurementList)
print("#"*150)


# hpf.printTargetList(tracker.__targetList__)
association = hpf.backtrackMeasurementsIndices(trackNodes)
print("Association",*association, sep = "\n")

fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
hpf.plotRadarOutline(p0, radarRange)
# hpf.plotVelocityArrowFromNode(trackNodes,2)
# hpf.plotValidationRegionFromNodes(trackNodes,sigma, 1)
# hpf.plotValidationRegionFromForest(tracker.__targetList__, sigma, 1)
hpf.plotMeasurementsFromForest(tracker.__targetList__, real = True, dummy = True)
# hpf.plotMeasurementsFromList(tracker.__scanHistory__)
# hpf.plotMeasurementsFromNodes(trackNodes)
# hpf.plotHypothesesTrack(tracker.__targetList__)
hpf.plotActiveTrack(trackNodes)
plt.axis("equal")
plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
plt.show()
print("Done :)")