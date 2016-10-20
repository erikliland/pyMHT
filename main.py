from classDefinitions import * 
import helpFunctions as hpf
import radarSimulator as sim
import matplotlib.pyplot as plt
import tomht

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

nScans = 5
simList = sim.simulateTargets(seed, initialTargets, nScans, tomht.timeStep, tomht.Phi(tomht.timeStep), tomht.Q, tomht.Gamma)
print("Sim list:")
print(*simList, sep = "\n", end = "\n\n")
scanList = sim.simulateScans(seed, simList, tomht.C, tomht.R, False, tomht.lambda_phi, radarRange)
# print("Scan list:")
# print(*scanList, sep = "\n", end = "\n\n")

for initialTarget in initialTargets:
 	tomht.initiateTarget(initialTarget)

associationHistory = []
for measurementIndex, measurementList in enumerate(scanList):
	associationHistory.append(tomht.addMeasurementList(measurementList))
	# if measurementIndex == 1:
	# 	break
# hpf.printTargetList(tomht.__targetList__)

association = hpf.backtrackMeasurementsIndices(associationHistory[-1])
print(*association, sep = "\n")

fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)
hpf.plotRadarOutline(p0, radarRange)
# hpf.plotVelocityArrowFromNode(associationHistory[-1],2)
# hpf.plotValidationRegionFromNodes(associationHistory[-1],tomht.sigma, 3)
hpf.plotMeasurementsFromForest(tomht.__targetList__, real = False, dummy = False)
hpf.plotMeasurementsFromList(tomht.__scanHistory__)
hpf.plotActiveTrack(associationHistory[-1])
plt.axis("equal")
plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
plt.show()
print("Done :)")