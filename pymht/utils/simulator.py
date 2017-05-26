import numpy as np
from pymht.utils import classDefinitions
from pymht.utils.classDefinitions import SimTarget, SimTargetCartesian
from pymht.utils.classDefinitions import MeasurementList, AIS_message, Position, AisMessagesList, SimList
import time
import copy
import math
import logging

log = logging.getLogger(__name__)

def checkEqualIvo(lst):
    return not lst or lst.count(lst[0]) == len(lst)

def seed_simulator(seed):
    np.random.seed(seed)

def generateInitialTargets(numOfTargets, centerPosition,
                           radarRange, P_d, sigma_Q, **kwargs):
    usedMMSI = []
    initialTime = time.time()
    initialList = []
    speeds = np.array([1, 10, 12, 15, 28, 35], dtype=np.float32) * 0.5  # ~knots to m/s
    for targetIndex in range(numOfTargets):
        heading = np.random.uniform(0, 360)
        distance = np.random.uniform(0, radarRange * 0.8)
        px, py = _pol2cart(heading, distance)
        px += centerPosition[0]
        py += centerPosition[1]
        heading = np.random.uniform(0, 360)
        speed = np.random.choice(speeds)
        vx, vy = _pol2cart(heading, speed)
        if kwargs.get('assignMMSI',False):
            while True:
                mmsi = np.random.randint(100000000,999999999)
                if mmsi not in usedMMSI:
                    usedMMSI.append(mmsi)
                    break
        else:
            mmsi = None
        target = SimTargetCartesian(np.array([px, py, vx, vy], dtype=np.float32), initialTime, P_d, sigma_Q, mmsi = mmsi)
        initialList.append(target)
    return initialList

def simulateTargets(initialTargets, simTime, timeStep, model, **kwargs):
    simList = SimList()
    assert all([isinstance(initialTarget,SimTarget) for initialTarget in initialTargets])
    simList.append(initialTargets)
    nTimeSteps = int(np.ceil(simTime / timeStep))

    for i in range(nTimeSteps):
        targetList = [target.calculateNextState(timeStep)
                      for target in simList[-1]]
        simList.append(targetList)

    return simList

def simulateScans(simList, radarPeriod, H, R, lambda_phi=0,
                  rRange=None, p0=None, **kwargs):
    includeInitialTime = not kwargs.get('preInitialized', False)
    area = np.pi * np.power(rRange, 2)
    gClutter = lambda_phi * area
    lClutter = kwargs.get('lambda_local', 1)
    scanList = []
    lastScan = None
    skippedFirst = False
    for targetList in simList:
        simTime = targetList[0].time
        if lastScan is None:
            if not includeInitialTime and not skippedFirst:
                skippedFirst = True
                lastScan = simTime
                continue
            lastScan = simTime
        else:
            timeSinceLastScan = simTime - lastScan
            if timeSinceLastScan >= radarPeriod:
                lastScan = simTime
            else:
                continue

        measurementList = MeasurementList(simTime)
        for target in targetList:
            visible = np.random.uniform() <= kwargs.get('P_d',target.P_d)
            if (rRange is not None) and (p0 is not None):
                inRange = target.inRange(p0, rRange)
            else:
                inRange = True

            if visible and inRange:
                measurementList.measurements.append(target.positionWithNoise())
                if kwargs.get('localClutter', True):
                    nClutter = np.random.poisson(lClutter)
                    log.debug("nLocalClutter {:}".format(nClutter))
                    measurementList.measurements.extend([target.positionWithNoise(sigma_R_scale = 3)
                                                         for _ in range(nClutter)])
        if all(e is not None for e in [rRange, p0]) and kwargs.get('globalClutter', True):
            nClutter = np.random.poisson(gClutter)
            log.debug("nGlobalClutter {:}".format(nClutter))
            for i in range(nClutter):
                clutter = _generateCartesianClutter(p0, rRange)
                measurementList.measurements.append(clutter)
        if kwargs.get("shuffle", True):
            np.random.shuffle(measurementList.measurements)
        nMeas = len(measurementList.measurements)
        measurementList.measurements = np.array(
            measurementList.measurements, ndmin=2, dtype=np.float32)
        measurementList.measurements = measurementList.measurements.reshape((nMeas, 2))
        scanList.append(copy.deepcopy(measurementList))
    return scanList

def simulateAIS(sim_list, ais_model, radarPeriod, initTime, **kwargs):
    ais_measurements = AisMessagesList()
    integerTime = kwargs.get('integerTime', True)
    tempList = []
    for i, sim in enumerate(sim_list[1:]):
        assert checkEqualIvo([t.time for t in sim])

        for j, target in ((j,target) for j, target in enumerate(sim) if target.mmsi is not None):
            if integerTime:
                messageTime = math.floor(target.time)
                dT = messageTime - target.time
                state = target.model.A_AIS(dT).dot(target.cartesianState())
            else:
                messageTime = target.time
                state = target.cartesianState()
            timeSinceLastAisMessage = messageTime - target.timeOfLastAisMessage
            speedMS = target.speedMS()
            reportingInterval = _aisReportInterval(speedMS, target.aisClass)
            shouldSendAisMessage = ((timeSinceLastAisMessage >= reportingInterval) and
                                    ((messageTime-initTime) % radarPeriod != 0))
            log.debug("MMSI " + str(target.mmsi) + " \t" +
                      "Target time " + str(target.time) + " \t" +
                      "Message time " + str(messageTime) + " \t" +
                      "Time of last AIS message " + str(target.timeOfLastAisMessage) + " \t" +
                      "Reporting Interval " + str(reportingInterval) + " \t" +
                      "Should send AIS message " + str(shouldSendAisMessage))
            if not shouldSendAisMessage:
                try:
                    sim_list[i + 2][j].timeOfLastAisMessage = target.timeOfLastAisMessage
                except IndexError:
                    pass
                continue
            try:
                sim_list[i+2][j].timeOfLastAisMessage = float(messageTime)
            except IndexError:
                pass
            highAccuracy = True
            if kwargs.get('noise', True):
                highAccuracy = np.random.uniform() > 0.5
                R = target.model.R_AIS(highAccuracy)
                v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
                state = target.model.C_AIS.dot(state) + v
                assert state.ndim == 1
                assert state.size == target.model.nObsDim_AIS, str(state.size)
            if kwargs.get('idScrambling',False) and np.random.uniform() > 0.5:
                mmsi = target.mmsi + 10
                log.info("Scrambling MMSI {0:} to {1:} at {2:}".format(target.mmsi,mmsi, messageTime))
            else:
                mmsi = target.mmsi

            prediction = AIS_message(time=messageTime,
                                     state=state,
                                     mmsi=mmsi,
                                     highAccuracy=highAccuracy)
            if np.random.uniform() <= target.P_r:
                tempList.append(prediction)
        simTime = sim[0].time
        if (simTime - initTime) % radarPeriod == 0:
            if tempList:
                ais_measurements.append(tempList[:])
                tempList = []
    return ais_measurements

def _aisReportInterval(speedMS, aisClass):
    from scipy.constants import knot
    speedKnot = speedMS * knot
    if aisClass.upper() == 'A':
        if speedKnot > 23:
            return 2
        if speedKnot > 14:
            return 4 #Should be 2 or 6, but are missing acceleration data
        if speedKnot > 0:
            return 6 #Should be 3.3 or 10, but are missing acceleration data
        if speedKnot == 0:
            return 60 #Should be 10s og 3min, but are missing mored status
        raise ValueError("Speed must be positive")
    elif aisClass.upper() == 'B':
        if speedKnot > 23:
            return 10
        if speedKnot > 14:
            return 5
        if speedKnot > 2:
            return 30
        if speedKnot >= 0:
            return 60*3
        raise ValueError("Speed must be positive")
    else:
        raise ValueError("aisClass must be 'A' og 'B'")

def findCenterPositionAndRange(simList):
    xMin = float('Inf')
    yMin = float('Inf')
    xMax = -float('Inf')
    yMax = -float('Inf')
    for sim in simList:
        for simTarget in sim:
            state = simTarget.cartesianState()
            xMin = state[0] if state[0] < xMin else xMin
            yMin = state[1] if state[1] < yMin else yMin
            xMax = state[0] if state[0] > xMax else xMax
            yMax = state[1] if state[1] > yMax else yMax
    p0 = Position(xMin + (xMax - xMin) / 2, yMin + (yMax - yMin) / 2)
    R = np.sqrt(np.power(max(abs(xMax - p0.x), abs(xMin - p0.x)), 2) +
                np.power(max(abs(yMax - p0.y), abs(yMin - p0.y)), 2))
    return p0, R

def _generateRadialClutter(centerPosition, radarRange):
    heading = np.random.uniform(0, 360)
    distance = np.random.uniform(0, radarRange)
    px, py = _pol2cart(heading, distance)
    return centerPosition + np.array([px, py])

def _generateCartesianClutter(centerPosition, radarRange):
    while True:
        x = np.random.uniform(-radarRange, radarRange)
        y = np.random.uniform(-radarRange, radarRange)
        pos = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(pos) <= radarRange:
            return centerPosition + pos

def _pol2cart(bearingDEG, distance):
    import math
    angleDEG = (90 - bearingDEG + 360)%360
    angleRAD = np.deg2rad(angleDEG)
    x = distance * math.cos(angleRAD)
    y = distance * math.sin(angleRAD)
    return [x, y]
