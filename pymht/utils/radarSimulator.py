import numpy as np
from pymht.utils.classDefinitions import *
import time


class SimTarget:

    def __init__(self, *args, **kwargs):
        p = kwargs.get('position')
        v = kwargs.get('velocity')
        t = kwargs.get('time')
        P_d = kwargs.get('P_d')
        if None not in [p, v, t, P_d]:
            self.state = np.array([p.x, p.y, v.x, v.y], dtype=np.double)
            self.time = t
            self.P_d = P_d
        elif len(args) == 2:
            self.state = args[0]
            self.time = args[1]
            self.P_d = None
        elif len(args) == 3:
            self.state = args[0]
            self.time = args[1]
            self.P_d = args[2]
        else:
            raise ValueError("Invalid arguments to SimTarget")

    def __str__(self):
        return ('Pos: ({0: 6.1f},{1: 6.1f})'.format(self.state[0], self.state[1]) + " " +
                'Vel: ({0: 6.1f},{1: 6.1f})'.format(self.state[2], self.state[3]) + " " +
                'Speed: {:4.1f}m/s'.format(self.speed()))

    def __repr__(self):
        return '({:.3e},{:.3e},{:.3e},{:.3e})'.format(*self.state)

    def storeString(self):
        return ',{0:.2f},{1:.2f}'.format(*self.state[0:2])

    def position(self):
        return Position(self.state[0], self.state[1])

    def velocity(self):
        return Velocity(self.state[2], self.state[3])

    def speed(self):
        return np.linalg.norm(self.state[2:4])

    def calculateNextState(self, timeStep, Phi, Q, Gamma):
        w = np.random.multivariate_normal(np.zeros(2), Q)
        nextState = Phi.dot(self.state) + Gamma.dot(w.T)
        return SimTarget(nextState, self.time + timeStep, self.P_d)

    def positionWithNoiseAndLoss(self, H,  R, P_d=1, p0=Position(0, 0), radarRange=None):
        if np.random.uniform() < P_d:
            v = np.random.multivariate_normal(np.zeros(2), R)
            return H.dot(self.state) + v
            # return Position(state[0], state[1])
        else:
            if radarRange is None:
                raise ValueError("If P_d < 1, p0 and radarRange is needed")
            return _generateCartesianClutter(p0, radarRange)


def generateInitialTargets(randomSeed, numOfTargets, centerPosition, radarRange, meanSpeed, P_d):
    np.random.seed(randomSeed)
    initialTime = time.time()
    initialList = []
    speeds = np.array([1, 10, 12, 15, 28, 35], dtype=np.double) * 0.5  # ~knots to m/s
    for targetIndex in range(numOfTargets):
        heading = np.random.uniform(0, 360)
        distance = np.random.uniform(0, radarRange * 0.8)
        px, py = _pol2cart(heading, distance)
        P0 = centerPosition + Position(px, py)
        heading = np.random.uniform(0, 360)
        speed = np.random.choice(speeds)
        vx, vy = _pol2cart(heading, speed)
        V0 = Velocity(vx, vy)
        target = SimTarget(np.array([px, py, vx, vy], dtype=np.double), initialTime, P_d)
        initialList.append(target)
    return initialList


def simulateTargets(randomSeed, initialTargets, numOfSteps, timeStep, Phi, Q, Gamma):
    np.random.seed(randomSeed)
    simList = []
    simList.append(initialTargets)
    for scan in range(numOfSteps):
        targetList = [target.calculateNextState(
            timeStep, Phi, Q, Gamma) for target in simList[-1]]
        simList.append(targetList)
    simList.pop(0)
    return simList


def simulateScans(randomSeed, simList, H, R, lambda_phi=None, rRange=None, p0=None, **kwargs):
    np.random.seed(randomSeed)
    area = np.pi * np.power(rRange, 2)
    lClutter = lambda_phi * area
    scanList = []
    for scan in simList:
        measurementList = MeasurementList(scan[0].time)
        measurementList.measurements = [target.positionWithNoiseAndLoss(
            H, R, target.P_d, p0, rRange) for target in scan]
        if (lambda_phi is not None) and (rRange is not None) and (p0 is not None):
            nClutter = np.random.poisson(lClutter)
            for i in range(nClutter):
                clutter = _generateCartesianClutter(p0, rRange)
                measurementList.measurements.append(clutter)
        if kwargs.get("shuffle", True):
            np.random.shuffle(measurementList.measurements)
        measurementList.measurements = np.array(
            measurementList.measurements, ndmin=2, dtype=np.double)
        scanList.append(measurementList)
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
                        SimTarget(	time=firstTime,
                                   position=initPos,
                                   velocity=(Position(elements[2 * i + 1], elements[2 * i + 2]) - initPos) * (1 / (localTime - firstTime))))

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
                target.state[2:4] = target.state[0:2] - \
                    firstPositions[targetIndex].toarray()
            elif scanIndex == (len(simList) - 1):
                target.state[2:4] = simList[scanIndex - 1][targetIndex].state[2:4]
            else:
                target.state[2:4] = target.state[0:2] - \
                    simList[scanIndex - 1][targetIndex].state[0:2]

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
        pos = np.array([x, y])
        if np.linalg.norm(pos) <= radarRange:
            return centerPosition.position + pos


def _pol2cart(bearingDEG, distance):
    angleDEG = 90 - bearingDEG
    angleRAD = np.deg2rad(angleDEG)
    x = distance * np.cos(angleRAD)
    y = distance * np.sin(angleRAD)
    return [x, y]
