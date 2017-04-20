import numpy as np
from pymht.utils.classDefinitions import SimTarget as Target
from pymht.utils.classDefinitions import MeasurementList, AIS_message, Position
import time
import copy
import math
import logging

# ----------------------------------------------------------------------------
# Instantiate logging object
# ----------------------------------------------------------------------------
log = logging.getLogger(__name__)


def positionWithNoise(state, H, R):
    assert R.ndim == 2
    assert R.shape[0] == R.shape[1]
    assert H.shape[1] == state.shape[0]
    v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
    assert H.shape[0] == v.shape[0], str(state.shape) + str(v.shape)
    assert v.ndim == 1
    return H.dot(state) + v


def calculateNextState(target, timeStep, Phi, Q, Gamma):
    Q_matrix = target.Q if target.Q is not None else Q
    w = np.random.multivariate_normal(np.zeros(2), Q_matrix)
    nextState = Phi.dot(target.state) + Gamma.dot(w.T)
    newVar = {'state': nextState, 'time': target.time + timeStep}
    return Target(**{**target.__dict__, **newVar})


def generateInitialTargets(randomSeed, numOfTargets, centerPosition,
                           radarRange, meanSpeed, P_d):
    np.random.seed(randomSeed)
    initialTime = time.time()
    initialList = []
    speeds = np.array([1, 10, 12, 15, 28, 35], dtype=np.float32) * 0.5  # ~knots to m/s
    for targetIndex in range(numOfTargets):
        heading = np.random.uniform(0, 360)
        distance = np.random.uniform(0, radarRange * 0.8)
        px, py = _pol2cart(heading, distance)
        heading = np.random.uniform(0, 360)
        speed = np.random.choice(speeds)
        vx, vy = _pol2cart(heading, speed)
        target = Target(np.array([px, py, vx, vy], dtype=np.float32), initialTime, P_d)
        initialList.append(target)
    return initialList


def simulateTargets(randomSeed, initialTargets, simTime, timeStep, Phi, Q, Gamma):
    np.random.seed(randomSeed)
    simList = []
    simList.append(initialTargets)
    nTimeSteps = int(simTime / timeStep)
    for i in range(nTimeSteps):
        targetList = [calculateNextState(target, timeStep, Phi, Q, Gamma)
                      for target in simList[-1]]
        simList.append(targetList)
    simList.pop(0)
    return simList


def simulateScans(randomSeed, simList, radarPeriod, H, R, lambda_phi=0,
                  rRange=None, p0=None, **kwargs):
    np.random.seed(randomSeed)
    area = np.pi * np.power(rRange, 2)
    gClutter = lambda_phi * area
    lClutter = kwargs.get('lClutter', 2)
    scanList = []
    lastScan = None
    for sim in simList:
        simTime = sim[0].time
        if lastScan is None:
            lastScan = simTime
        else:
            timeSinceLastScan = simTime - lastScan
            if timeSinceLastScan >= radarPeriod:
                lastScan = simTime
            else:
                continue

        measurementList = MeasurementList(simTime)
        for target in sim:
            visible = np.random.uniform() <= target.P_d
            if (rRange is not None) and (p0 is not None):
                distance = np.linalg.norm(target.state[0:2] - p0.array)
                inRange = distance <= rRange
            else:
                inRange = True

            if visible and inRange:
                measurementList.measurements.append(positionWithNoise(target.state, H, R))
                if kwargs.get('localClutter'):
                    nClutter = np.random.poisson(lClutter)
                    log.debug("nLocalClutter {:}".format(nClutter))
                    measurementList.measurements.extend([positionWithNoise(target.state, H, R * 5)
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


def simulateAIS(random_seed, sim_list, Phi_func, C, R, P_0, **kwargs):
    np.random.seed(random_seed)
    ais_measurements = []
    integerTime = kwargs.get('integerTime', True)
    aisPeriod = kwargs.get('period', 5.0)
    prevTime = sim_list[0][0].time
    for sim in sim_list[1:]:
        if not (sim[0].time - prevTime > aisPeriod): continue
        tempList = []
        for target in (t for t in sim if t.mmsi is not None):
            if integerTime:
                time = math.floor(target.time)
                dT = time - target.time
                state = Phi_func(dT).dot(target.state)
            else:
                time = target.time
                state = target.state
            if kwargs.get('noise', True):
                state = positionWithNoise(state, C, R)
            if kwargs.get('idScrambling',False) and np.random.uniform() > 0.5:
                mmsi = target.mmsi + 10
                log.info("Scrambling MMSI {0:} to {1:} at {2:}".format(target.mmsi,mmsi, time))
            else:
                mmsi = target.mmsi

            prediction = AIS_message(time=time,
                                     state=state,
                                     covariance=P_0,
                                     mmsi=mmsi)
            tempList.append(prediction)
        if tempList:
            ais_measurements.append(tempList)
        prevTime = sim[0].time
    return ais_measurements


def writeSimList(initialTargets, simList, filename):
    startTime = initialTargets[0].time
    try:
        f = open(filename, 'w')
        f.write("0" + "".join([target.storeString() for target in initialTargets]) + "\n")
        for scan in simList:
            f.write('{:.2f}'.format(scan[0].time - startTime) +
                    "".join([target.storeString() for target in scan]) + "\n")
        f.close()
    except:
        pass


def importFromFile(filename, **kwargs):
    startLine = kwargs.get('startLine', 0)
    initialTime = time.time()
    initialTargets = []
    simList = []
    firstPositions = None
    firstTime = None
    try:
        f = open(filename, 'r')
    except:
        print("Could not open the file:", filename)
        return [], []
    for lineIndex, line in enumerate(f):
        lineIndex = lineIndex - startLine
        elements = line.strip().split(',')
        localTime = float(elements[0])
        globalTime = initialTime + localTime
        if lineIndex == 0:
            firstTime = float(elements[0])
            firstPositions = [Position(elements[i], elements[i + 1])
                              for i in range(1, len(elements), 2)]
        elif lineIndex > 0:
            if lineIndex == 1:
                for i, initPos in enumerate(firstPositions):
                    initialTargets.append(
                        Target(time=firstTime,
                               position=initPos,
                               velocity=(Position(elements[2 * i + 1], elements[2 * i + 2]) - initPos) * (
                                   1 / (localTime - firstTime))))

            if localTime.is_integer():
                targetList = [Target(time=localTime,
                                     position=Position(elements[i], elements[i + 1]),
                                     velocity=Velocity(0, 0)
                                     ) for i in range(1, len(elements), 2)
                              ]

                simList.append(targetList)
    for scanIndex, scan in enumerate(simList):
        for targetIndex, target in enumerate(scan):
            if scanIndex == 0:
                target.state[2:4] = (target.state[0:2] -
                                     firstPositions[targetIndex].toarray())
            elif scanIndex == (len(simList) - 1):
                target.state[2:4] = simList[scanIndex - 1][targetIndex].state[2:4]
            else:
                target.state[2:4] = (target.state[0:2] -
                                     simList[scanIndex - 1][targetIndex].state[0:2])

    return initialTargets, simList


def findCenterPositionAndRange(simList):
    xMin = float('Inf')
    yMin = float('Inf')
    xMax = -float('Inf')
    yMax = -float('Inf')
    for sim in simList:
        for simTarget in sim:
            xMin = simTarget.state[0] if simTarget.state[0] < xMin else xMin
            yMin = simTarget.state[1] if simTarget.state[1] < yMin else yMin
            xMax = simTarget.state[0] if simTarget.state[0] > xMax else xMax
            yMax = simTarget.state[1] if simTarget.state[1] > yMax else yMax
    p0 = Position(xMin + (xMax - xMin) / 2, yMin + (yMax - yMin) / 2)
    R = np.sqrt(np.power(max(abs(xMax - p0.x), abs(xMin - p0.x)), 2) +
                np.power(max(abs(yMax - p0.y), abs(yMin - p0.y)), 2))
    return p0, R


def _generateRadialClutter(centerPosition, radarRange):
    heading = np.random.uniform(0, 360)
    distance = np.random.uniform(0, radarRange)
    px, py = _pol2cart(heading, distance)
    return centerPosition.array + np.array([px, py])


def _generateCartesianClutter(centerPosition, radarRange):
    while True:
        x = np.random.uniform(-radarRange, radarRange)
        y = np.random.uniform(-radarRange, radarRange)
        pos = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(pos) <= radarRange:
            return centerPosition.array + pos


def _pol2cart(bearingDEG, distance):
    angleDEG = 90 - bearingDEG
    angleRAD = np.deg2rad(angleDEG)
    x = distance * np.cos(angleRAD)
    y = distance * np.sin(angleRAD)
    return [x, y]
