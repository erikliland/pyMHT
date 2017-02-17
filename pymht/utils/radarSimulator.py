import numpy as np
from pymht.utils.classDefinitions import *
import time
import copy


def positionWithNoise(state, H, R):
    v = np.random.multivariate_normal(np.zeros(2), R)
    return H.dot(state) + v


def calculateNextState(target, timeStep, Phi, Q, Gamma):
    w = np.random.multivariate_normal(np.zeros(2), Q)
    nextState = Phi.dot(target.state) + Gamma.dot(w.T)
    return Target(nextState, target.time + timeStep, target.P_d)


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
        P0 = centerPosition + Position(px, py)
        heading = np.random.uniform(0, 360)
        speed = np.random.choice(speeds)
        vx, vy = _pol2cart(heading, speed)
        V0 = Velocity(vx, vy)
        target = Target(np.array([px, py, vx, vy], dtype=np.float32), initialTime, P_d)
        initialList.append(target)
    return initialList


def simulateTargets(randomSeed, initialTargets, numOfSteps, timeStep, Phi, Q, Gamma):
    np.random.seed(randomSeed)
    simList = []
    simList.append(initialTargets)
    for scan in range(numOfSteps):
        targetList = [calculateNextState(target, timeStep, Phi, Q, Gamma)
                      for target in simList[-1]]
        simList.append(targetList)
    simList.pop(0)
    return simList


def simulateScans(randomSeed, simList, H, R, lambda_phi=None,
                  rRange=None, p0=None, **kwargs):
    np.random.seed(randomSeed)
    area = np.pi * np.power(rRange, 2)
    lClutter = lambda_phi * area
    scanList = []
    for scan in simList:
        measurementList = MeasurementList(scan[0].time)
        measurementList.measurements = []  # BUG: Why is this neccesary
        for target in scan:
            visible = np.random.uniform() <= target.P_d
            if (rRange is not None) and (p0 is not None):
                distance = np.linalg.norm(target.state[0:2] - p0.position)
                inRange = distance <= rRange
            else:
                inRange = True

            if visible and inRange:
                measurementList.measurements.append(positionWithNoise(target.state, H, R))
        if (lambda_phi is not None) and (rRange is not None) and (p0 is not None):
            nClutter = np.random.poisson(lClutter)
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
    return centerPosition + Position(px, py)


def _generateCartesianClutter(centerPosition, radarRange):
    while True:
        x = np.random.uniform(-radarRange, radarRange)
        y = np.random.uniform(-radarRange, radarRange)
        pos = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(pos) <= radarRange:
            return centerPosition.position + pos


def _pol2cart(bearingDEG, distance):
    angleDEG = 90 - bearingDEG
    angleRAD = np.deg2rad(angleDEG)
    x = distance * np.cos(angleRAD)
    y = distance * np.sin(angleRAD)
    return [x, y]
