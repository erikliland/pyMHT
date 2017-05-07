import numpy as np
from pymht.utils.classDefinitions import SimTarget
from pymht.utils.classDefinitions import MeasurementList, AIS_message, Position, AIS_messageList, SimList
import time
import copy
import math
import logging

# ----------------------------------------------------------------------------
# Instantiate logging object
# ----------------------------------------------------------------------------
log = logging.getLogger(__name__)

def seed_simulator(seed):
    np.random.seed(seed)

def positionWithNoise(state, H, R):
    assert R.ndim == 2
    assert R.shape[0] == R.shape[1]
    assert H.shape[1] == state.shape[0]
    v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
    assert H.shape[0] == v.shape[0], str(state.shape) + str(v.shape)
    assert v.ndim == 1
    return H.dot(state) + v


def calculateNextState(target, timeStep, Phi, model):
    Q = model.Q(timeStep, target.sigma_Q)
    w = np.random.multivariate_normal(np.zeros(4), Q)
    nextState = Phi.dot(target.state) + w.T
    newVar = {'state': nextState, 'time': target.time + timeStep}
    return SimTarget(**{**target.__dict__, **newVar})


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
        target = SimTarget(np.array([px, py, vx, vy], dtype=np.float32), initialTime, P_d, sigma_Q, mmsi = mmsi)
        initialList.append(target)
    return initialList


def simulateTargets(initialTargets, simTime, timeStep, model, **kwargs):
    Phi = model.Phi(timeStep)
    simList = SimList()
    assert all([type(initialTarget) == SimTarget for initialTarget in initialTargets])
    simList.append(initialTargets)
    nTimeSteps = int(simTime / timeStep)
    for i in range(nTimeSteps):
        targetList = [calculateNextState(target, timeStep, Phi, model)
                      for target in simList[-1]]
        simList.append(targetList)
    if not kwargs.get('includeInitialTime', True):
        simList.pop(0)
    return simList


def simulateScans(simList, radarPeriod, H, R, lambda_phi=0,
                  rRange=None, p0=None, **kwargs):
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
            visible = np.random.uniform() <= kwargs.get('P_d',target.P_d)
            if (rRange is not None) and (p0 is not None):
                distance = np.linalg.norm(target.state[0:2] - p0)
                inRange =  distance <= rRange
                inRange = target.inRange(p0, rRange)
            else:
                inRange = True

            if visible and inRange:
                measurementList.measurements.append(positionWithNoise(target.state, H, R))
                if kwargs.get('localClutter', True):
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


def simulateAIS(sim_list, Phi_func, C, R, P_0, radarPeriod, initTime, **kwargs):
    ais_measurements = AIS_messageList()
    integerTime = kwargs.get('integerTime', True)
    for i, sim in enumerate(sim_list[1:]):
        tempList = []
        for j, target in ((j,t) for j, t in enumerate(sim) if t.mmsi is not None):
            if integerTime:
                messageTime = math.floor(target.time)
                dT = messageTime - target.time
                state = Phi_func(dT).dot(target.state)
            else:
                messageTime = target.time
                state = target.state
            timeSinceLastAisMessage = target.time - target.timeOfLastAisMessage
            speedMS = np.linalg.norm(target.state[2:4])
            reportingInterval = _aisReportInterval(speedMS, target.aisClass)
            shouldSendAisMessage = ((timeSinceLastAisMessage >= reportingInterval) and
                                    ((messageTime-initTime) % radarPeriod != 0))
            log.debug("MMSI " + str(target.mmsi) +
                  "Time " + str(target.time) + " \t" +
                  "Time of last AIS message " + str(target.timeOfLastAisMessage) + " \t" +
                  "Reporting Interval " + str(reportingInterval) +
                  "Should send AIS message " + str(shouldSendAisMessage))
            if not shouldSendAisMessage:
                try:
                    sim_list[i + 2][j].timeOfLastAisMessage = target.timeOfLastAisMessage
                except IndexError:
                    pass
                continue
            try:
                sim_list[i+2][j].timeOfLastAisMessage = target.time
            except IndexError:
                pass

            if kwargs.get('noise', True):
                state = positionWithNoise(state, C, R)
            if kwargs.get('idScrambling',False) and np.random.uniform() > 0.5:
                mmsi = target.mmsi + 10
                log.info("Scrambling MMSI {0:} to {1:} at {2:}".format(target.mmsi,mmsi, messageTime))
            else:
                mmsi = target.mmsi

            prediction = AIS_message(time=messageTime,
                                     state=state,
                                     covariance=P_0,
                                     mmsi=mmsi)
            if np.random.uniform() <= target.P_r:
                tempList.append(prediction)
        if tempList:
            ais_measurements.append(tempList)
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
                        SimTarget(time=firstTime,
                                  position=initPos,
                                  velocity=(Position(elements[2 * i + 1], elements[2 * i + 2]) - initPos) * (
                                   1 / (localTime - firstTime))))

            if localTime.is_integer():
                targetList = [SimTarget(time=localTime,
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
    return centerPosition + np.array([px, py])


def _generateCartesianClutter(centerPosition, radarRange):
    while True:
        x = np.random.uniform(-radarRange, radarRange)
        y = np.random.uniform(-radarRange, radarRange)
        pos = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(pos) <= radarRange:
            return centerPosition + pos


def _pol2cart(bearingDEG, distance):
    angleDEG = 90 - bearingDEG
    angleRAD = np.deg2rad(angleDEG)
    x = distance * np.cos(angleRAD)
    y = distance * np.sin(angleRAD)
    return [x, y]

