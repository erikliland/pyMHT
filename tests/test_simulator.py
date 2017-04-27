# content of test_sample.py
import pymht.utils.simulator as sim
import numpy as np
import pymht.models.pv as model
from pymht.utils.classDefinitions import AIS_message

seed = 172362
nTargets = 10
radarPeriod = 60./24.
simTime = 10
simulationTimeStep = radarPeriod/2
p0 = np.array([0, 0])
radarRange = 1000
P_d = 0.8
lambda_phi = 1e-6
aisPeriod = radarPeriod
nTests = 10
initialTargets = None
simLists = None
aisMeasurements = None
scanList = None

sim.seed_simulator(seed)


def test_initial_target_generation():
    global initialTargets
    initialTargets = sim.generateInitialTargets(nTargets, p0, radarRange, P_d,model.sigmaQ_true, assignMMSI = True)

def test_simulation_seed_consistency():
    global simLists
    simLists = []
    for i in range(nTests):
        sim.seed_simulator(seed)
        simLists.append(sim.simulateTargets(initialTargets,
                                              simTime,
                                              simulationTimeStep,
                                              model))

    for i in range(nTests-1):
        for simListA, simListB in zip(simLists[i],simLists[i+1]):
            for targetA, targetB in zip(simListA, simListB):
                assert targetA == targetB

def test_scan_simulation_consistency():
    global scanList
    scanLists = []
    for _ in range(nTests):
        sim.seed_simulator(seed)
        scanLists.append(sim.simulateScans(simLists[0],
                                     radarPeriod,
                                     model.C_RADAR,
                                     model.R_RADAR(model.sigmaR_RADAR_true),
                                     lambda_phi,
                                     radarRange,
                                     p0,
                                     shuffle=True,
                                     localClutter=True,
                                     globalClutter=True)
                         )
    for i in range(nTests-1):
        scanListA = scanLists[i]
        scanListB = scanLists[i+1]
        for measurementListA, measurementListB in zip(scanListA, scanListB):
            assert measurementListA == measurementListB

def test_ais_simulation_consistency():
    """
    Known issue: when using integer time, different initial (target) time, 
    will give different AIS results depending on the decimal time at init.
    :return: 
    """
    global aisMeasurements
    aisMeasurementsList = []
    for i in range(nTests):
        sim.seed_simulator(seed)
        aisMeasurementsList.append(
            sim.simulateAIS(simLists[i],
                          model.Phi,
                          model.C_AIS,
                          model.R_AIS(model.sigmaR_AIS_true),
                          model.GPS_COVARIANCE_PRECISE,
                          radarPeriod,
                          integerTime=True,
                          noise=True,
                          idScrambling = False,
                          period=aisPeriod))


    for i in range(nTests-1):
        simA = aisMeasurementsList[i]
        simB = aisMeasurementsList[i+1]
        for listA, listB in zip(simA,simB):
            for messageA, messageB in zip(listA, listB):
                assert type(messageA) == AIS_message
                assert messageA == messageB


if __name__ == '__main__':
    test_initial_target_generation()
    test_simulation_seed_consistency()
    test_scan_simulation_consistency()
    test_ais_simulation_consistency()