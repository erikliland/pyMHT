import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import logging
import copy
import collections
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from . import helpFunctions as hpf
from .xmlDefinitions import *
from ..models import pv, polar, ais
log = logging.getLogger(__name__)


class SimTarget:

    def __init__(self, state, time, P_d, sigma_Q, **kwargs):
        self.state = np.array(state, dtype=np.double)
        assert self.state.ndim == 1
        self.time = time
        self.P_d = P_d
        self.sigma_Q = sigma_Q
        self.mmsi = kwargs.get('mmsi')
        self.aisClass = kwargs.get('aisClass', 'B')
        self.timeOfLastAisMessage = kwargs.get('timeOfLastAisMessage', -float('inf'))
        self.P_r = kwargs.get('P_r', 1.)
        self.model = None

    def __eq__(self, other):
        if not np.array_equal(self.state, other.state):
            return False
        if self.time != other.time:
            return False
        if self.P_d != other.P_d:
            return False
        if self.mmsi != other.mmsi:
            return False
        if self.sigma_Q != other.sigma_Q:
            return False
        if self.P_r != other.P_r:
            return False
        return True

    def inRange(self, p0, rRange):
        distance = np.linalg.norm(self.state[0:2] - p0)
        return distance <= rRange

    def storeString(self):
        return ',{0:.2f},{1:.2f}'.format(*self.state[0:2])

    def getXmlStateStringsCartesian(self, precision=2):
        raise NotImplementedError

    def getXmlStateStringsPolar(self, precision=2):
        raise NotImplementedError

    def position(self):
        return Position(self.state[0], self.state[1])

    def velocity(self):
        raise NotImplementedError

    def speedMS(self):
        raise NotImplementedError

    def speedKnots(self):
        return self.speedMS() * 1.94384449

    def cartesianState(self):
        raise NotImplementedError

    def cartesianVelocity(self):
        raise NotImplementedError

    def polarState(self):
        raise NotImplementedError

    def calculateNextState(self, timeStep):
        raise NotImplementedError

    def positionWithNoise(self):
        raise NotImplementedError


class SimTargetCartesian(SimTarget):

    def __init__(self, state, time, P_d, sigma_Q, **kwargs):
        SimTarget.__init__(self, state, time, P_d, sigma_Q, **kwargs)
        self.model = pv

    def __str__(self):
        timeString = datetime.datetime.fromtimestamp(self.time).strftime("%H:%M:%S.%f")
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        return ('Time: ' + timeString + " " +
                'Pos: ({0: 7.1f},{1: 7.1f})'.format(self.state[0], self.state[1]) + " " +
                'Vel: ({0: 5.1f},{1: 5.1f})'.format(self.state[2], self.state[3]) + " " +
                'Speed: {0:4.1f}m/s ({1:4.1f}knt)'.format(self.speedMS(), self.speedKnots()) + " " +
                'Pd: {:3.0f}%'.format(self.P_d * 100.) + " " +
                mmsiString)

    def __copy__(self):
        return SimTargetCartesian(self.state, self.time, self.P_d, self.sigma_Q,
                                  mmsi=self.mmsi, aisClass=self.aisClass, timeOfLastAisMessage=self.timeOfLastAisMessage,
                                  P_r=self.P_r)

    __repr__ = __str__

    def getXmlStateStringsCartesian(self, precision=2):
        return (str(round(self.state[0], precision)),
                str(round(self.state[1], precision)),
                str(round(self.state[2], precision)),
                str(round(self.state[3], precision)))

    def position(self):
        return Position(self.state[0], self.state[1])

    def velocity(self):
        return Velocity(self.state[2], self.state[3])

    def speedMS(self):
        speed_ms = np.linalg.norm(self.state[2:4])
        return speed_ms

    def cartesianState(self):
        return self.state

    def cartesianVelocity(self):
        return self.state[2:4]

    def calculateNextState(self, timeStep):
        Phi = self.model.Phi(timeStep)
        Q = self.model.Q(timeStep, self.sigma_Q)
        w = np.random.multivariate_normal(np.zeros(4), Q)
        nextState = Phi.dot(self.state) + w.T
        newVar = {'state': nextState, 'time': self.time + timeStep}
        return SimTargetCartesian(**{**self.__dict__, **newVar})

    def positionWithNoise(self, **kwargs):
        # def positionWithNoise(state, H, R):
        sigma_R_scale = kwargs.get('sigma_R_scale', 1)
        R = self.model.R_RADAR(self.model.sigmaR_RADAR_true * sigma_R_scale)
        H = self.model.C_RADAR
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        assert H.shape[1] == self.state.shape[0]
        v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        assert H.shape[0] == v.shape[0], str(self.state.shape) + str(v.shape)
        assert v.ndim == 1
        return H.dot(self.state) + v


class SimTargetPolar(SimTarget):

    def __init__(self, state, time, P_d, sigma_Q, **kwargs):
        SimTarget.__init__(self, state, time, P_d, sigma_Q, **kwargs)
        self.model = polar
        self.headingChangeMean = kwargs.get('headingChangeMean')
        #state = east, north, heading (deg), speed

    def __str__(self):
        timeString = datetime.datetime.fromtimestamp(self.time).strftime("%H:%M:%S.%f")
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        return ('Time: ' + timeString + " " +
                'Pos: ({0: 7.1f},{1: 7.1f})'.format(self.state[0], self.state[1]) + " " +
                'Hdg: {0:5.1f}{1:+3.1f} deg'.format(self.state[2], self.headingChangeMean) + " " +
                'Speed: {0:4.1f}m/s ({1:4.1f}knt)'.format(self.speedMS(), self.speedKnots()) + " " +
                'Pd: {:3.0f}%'.format(self.P_d * 100.) + " " +
                mmsiString)

    def __copy__(self):
        return SimTargetPolar(self.state, self.time, self.P_d, self.sigma_Q,
                              mmsi=self.mmsi, aisClass=self.aisClass, timeOfLastAisMessage=self.timeOfLastAisMessage,
                              P_r=self.P_r, headingChangeMean=self.headingChangeMean)

    __repr__ = __str__

    def getXmlStateStringsCartesian(self, precision=2):
        cartesianState = self.cartesianState()
        return (str(round(cartesianState[0], precision)),
                str(round(cartesianState[1], precision)),
                str(round(cartesianState[2], precision)),
                str(round(cartesianState[3], precision)))

    def getXmlStateStringsPolar(self, precision=2):
        return (str(round(self.state[0], precision)),
                str(round(self.state[1], precision)),
                str(round(self.state[2], precision)),
                str(round(self.state[3], precision)))

    @staticmethod
    def normalizeHeadingDeg(heading):
        return (heading + 360.) % 360.

    def cartesianState(self):
        pos = self.state[0:2]
        vel = self.cartesianVelocity()
        return np.hstack((pos, vel))

    def cartesianVelocity(self):
        compassHeadingDeg = self.state[2]
        theta = math.radians(self.normalizeHeadingDeg(90. - compassHeadingDeg))
        vx = self.state[3] * math.cos(theta)
        vy = self.state[3] * math.sin(theta)
        return np.array([vx, vy], dtype=np.float32)

    def position(self):
        return Position(self.state[0], self.state[1])

    def velocity(self):
        return Velocity(self.cartesianVelocity())

    def speedMS(self):
        return self.state[3]

    def calculateNextState(self, timeStep):
        cartesianSpeedVector = self.cartesianVelocity()
        stateDelta = timeStep * cartesianSpeedVector
        headingDelta = timeStep * np.random.normal(self.headingChangeMean, self.model.sigma_hdg)
        speedDelta = timeStep * np.random.normal(0, self.model.sigma_speed)
        nextState = np.copy(self.state)
        nextState[0:2] += stateDelta
        nextState[2] = self.normalizeHeadingDeg(nextState[2] + headingDelta)
        nextState[3] = max(0, nextState[3] + speedDelta)
        newVar = {'state': nextState, 'time': self.time + timeStep}
        return SimTargetPolar(**{**self.__dict__, **newVar})

    def positionWithNoise(self, **kwargs):
        sigma_R_scale = kwargs.get('sigma_R_scale', 1)
        R = self.model.R_RADAR(self.model.sigmaR_RADAR_true * sigma_R_scale)
        H = self.model.C_RADAR
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        assert H.shape[1] == self.state.shape[0]
        v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        assert H.shape[0] == v.shape[0], str(self.state.shape) + str(v.shape)
        assert v.ndim == 1
        return H.dot(self.state) + v


class Position:

    def __init__(self, *args, **kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        if (x is not None) and (y is not None):
            self.array = np.array([x, y])
        elif len(args) == 1:
            self.array = np.array(args[0])
        elif len(args) == 2:
            self.array = np.array([args[0], args[1]])
        else:
            raise ValueError("Invalid arguments to Position")

    def __str__(self):
        return 'Pos: ({0: 8.2f},{1: 8.2f})'.format(self.array[0], self.array[1])

    def __repr__(self):
        return '({0:.3e},{1:.3e})'.format(self.array[0], self.array[1])

    def __add__(self, other):
        return Position(self.array + other.position)

    def __sub__(self, other):
        return Position(self.array - other.position)

    def __mul__(self, other):
        return Position(self.array * other.position)

    def __div__(self, other):
        return Position(self.array / other.position)

    def x(self):
        return self.array[0]

    def y(self):
        return self.array[1]

    def plot(self, ax=plt.gca(), measurementNumber=-1, scanNumber=None, mmsi=None, **kwargs):
        if mmsi is not None:
            marker = 'h' if kwargs.get('original', False) else 'D'
            ax.plot(self.array[0], self.array[1],
                    marker=marker, markerfacecolor='None',
                    markeredgewidth=kwargs.get('markeredgewidth', 1),
                    markeredgecolor=kwargs.get('color', 'black'))
        elif measurementNumber > 0:
            ax.plot(self.array[0], self.array[1], 'kx',
                    markeredgecolor=kwargs.get('color', 'black'))
        elif measurementNumber == 0:
            ax.plot(self.array[0], self.array[1], fillstyle="none", marker="o",
                    markeredgecolor=kwargs.get('color', 'black'))
        else:
            raise ValueError("Not a valid measurement number")

        if ((scanNumber is not None) and
                (measurementNumber is not None) and
                kwargs.get("labels", False)):
            ax.text(self.array[0], self.array[1], str(
                scanNumber) + ":" + str(measurementNumber), size=7, ha="left", va="top")


class Velocity:

    def __init__(self, *args, **kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        if (x is not None) and (y is not None):
            self.velocity[0] = np.array([x, y])
        elif len(args) == 1:
            self.velocity = np.array(args[0])
        elif len(args) == 2:
            self.velocity = np.array(args[0], args[1])
        else:
            raise ValueError("Invalid arguments to Velocity")

    def __str__(self):
        return 'Vel: ({: 6.2f},{: 6.2f})'.format(self.velocity[0], self.velocity[1])

    def __repr__(self):
        return '({:.3e},{:.3e})'.format(self.velocity[0], self.velocity[1])

    def __add__(self, other):
        return Velocity(self.velocity + other.velocity)

    def __sub__(self, other):
        return Velocity(self.velocity - other.velocity)

    def __mul__(self, other):
        return Velocity(self.velocity * other.velocity)

    def __div__(self, other):
        return Velocity(self.velocity / other.velocity)

    def x(self):
        return self.velocity[0]

    def y(self):
        return self.velocity[1]


class SimList(list):

    def __init__(self, *args):
        list.__init__(self, *args)

    def storeGroundTruth(self, scenarioElement, scenario, **kwargs):
        if self is None:
            return
        nSamples = len(self)
        nTargets = len(self[0])
        p0 = scenario.p0
        radarRange = scenario.radarRange
        radarPeriod = scenario.radarPeriod
        initialTime = scenario.initTime
        groundtruthElement = ET.SubElement(scenarioElement, groundtruthTag)
        for i in range(nTargets):
            trackElement = ET.SubElement(groundtruthElement,
                                         trackTag,
                                         attrib={idTag: str(i)})
            statesElement = ET.SubElement(trackElement,
                                          statesTag)
            sampleCounter = 0
            for j in range(nSamples):
                simTarget = self[j][i]
                inRange = simTarget.inRange(p0, radarRange)
                radarTime = ((simTarget.time - initialTime) % radarPeriod) == 0.
                if (not inRange) or (not radarTime):
                    continue
                sampleCounter += 1
                stateElement = ET.SubElement(statesElement,
                                             stateTag,
                                             attrib={timeTag: str(simTarget.time),
                                                     pdTag: str(simTarget.P_d)})
                eastPos, northPos, eastVel, northVel = simTarget.getXmlStateStringsCartesian()
                positionElement = ET.SubElement(stateElement, positionTag)
                ET.SubElement(positionElement, northTag).text = northPos
                ET.SubElement(positionElement, eastTag).text = eastPos
                velocityElement = ET.SubElement(stateElement, velocityTag)
                ET.SubElement(velocityElement, northTag).text = northVel
                ET.SubElement(velocityElement, eastTag).text = eastVel
                if simTarget.mmsi is not None:
                    trackElement.attrib[mmsiTag] = str(simTarget.mmsi)
                    trackElement.attrib[aisclassTag] = str(simTarget.aisClass)
                    trackElement.attrib[prTag] = str(simTarget.P_r)
                statesElement.attrib[sigmaqTag] = str(simTarget.sigma_Q)
                trackElement.attrib[lengthTag] = str(sampleCounter)

    def plot(self, ax=plt.gca(), **kwargs):
        colors = kwargs.get("colors")
        newArgs = copy.copy(kwargs)
        if "colors" in newArgs:
            del newArgs["colors"]

        nScan = len(self)
        nTargets = len(self[0])
        stateArray = np.zeros((nScan, nTargets, 4))
        for row, targetList in enumerate(self):
            stateArray[row, :, :] = np.array([target.cartesianState() for target in targetList])
        for col in range(nTargets):
            ax.plot(stateArray[:, col, 0],
                    stateArray[:, col, 1],
                    '.',
                    alpha=kwargs.get('alpha', 0.7),
                    markeredgewidth=kwargs.get('markeredgewidth', 0.5),
                    color=next(colors) if colors is not None else None,
                    markevery=kwargs.get('markevery', 1))

        for col, target in enumerate(self[0]):
            if kwargs.get('markStart', True):
                ax.plot(stateArray[0, col, 0], stateArray[0, col, 1], '.', color='black')
            if kwargs.get('label', False):
                velocity = target.cartesianVelocity()
                normVelocity = (velocity /
                                np.linalg.norm(velocity))
                offsetScale = kwargs.get('offset', 0.0)
                offset = offsetScale * np.array(normVelocity)
                position = stateArray[0, col, 0:2] - offset
                (horizontalalignment,
                 verticalalignment) = hpf._getBestTextPosition(normVelocity)
                ax.text(position[0],
                        position[1],
                        "T" + str(col),
                        fontsize=kwargs.get('fontsize', 10),
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment)


class AIS_message:

    def __init__(self, time, state, mmsi, highAccuracy=False):
        self.time = time
        self.state = state
        self.mmsi = mmsi
        self.highAccuracy = highAccuracy

    def __str__(self):
        timeString = self.getTimeString()
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        return ('Time: ' + timeString + " " +
                'State:' + np.array2string(self.state, formatter={'float_kind': lambda x: '{: 7.1f}'.format(x)}) + " " +
                'High accuracy: {:1} '.format(self.highAccuracy) +
                mmsiString)

    def __eq__(self, other):
        if self.time != other.time:
            return False
        if not np.array_equal(self.state, other.state):
            return False
        if not np.array_equal(self.covariance, other.covariance):
            return False
        if self.mmsi != other.mmsi:
            return False
        return True

    __repr__ = __str__

    def getTimeString(self):
        if self.time == int(self.time):
            timeFormat = "%H:%M:%S"
        else:
            timeFormat = "%H:%M:%S.%f"
        timeString = datetime.datetime.fromtimestamp(self.time).strftime(timeFormat)
        if self.time < 1e6:
            timeString = str(self.time)
        return timeString

    def plot(self, ax=plt.gca(), **kwargs):
        Position(self.state[0:2]).plot(ax, mmsi=self.mmsi, original=True, **kwargs)

    def predict(self, dT):
        Phi = ais.Phi(dT)
        Q = pv.Q(dT)
        state = Phi.dot(self.state)
        covariance = Phi.dot(pv.P0).dot(Phi.T) + Q
        return state, covariance


class AIS_prediction:

    def __init__(self, state, covariance, mmsi):
        assert state.shape[0] == covariance.shape[0] == covariance.shape[1]
        assert type(mmsi) is int
        self.state = state
        self.covariance = covariance
        self.mmsi = mmsi

    def __str__(self):
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        stateString = np.array_str(self.state, precision=1)
        covarianceString = 'Covariance diagonal: ' + np.array_str(np.diagonal(self.covariance),
                                                                  precision=1, suppress_small=True)
        return (stateString + " " + covarianceString + " " + mmsiString)

    __repr__ = __str__


class AisMessagesList:

    def __init__(self, *args):
        self._list = list(*args)
        self._lastExtractedTime = None
        self._iterator = None
        self._nextAisMeasurements = None

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __iter__(self):
        return self._list.__iter__()

    def append(self, *args):
        self._list.append(*args)

    def pop(self, *args):
        self._list.pop(*args)

    def print(self):
        print("aisMeasurements:")
        for aisTimeList in self._list:
            print(*aisTimeList, sep="\n", end="\n\n")

    def getMeasurements(self, scanTime):
        if self._iterator is None:
            self._iterator = (m for m in self._list)
            self._nextAisMeasurements = next(self._iterator, None)

        if self._nextAisMeasurements is not None:
            if all((m.time <= scanTime) for m in self._nextAisMeasurements):
                self._lastExtractedTime = scanTime
                res = AisMessageList(copy.copy(self._nextAisMeasurements))
                self._nextAisMeasurements = next(self._iterator, None)
                return res
        return AisMessageList()

    def predictAisMeasurements(self, scanTime, aisMeasurements):
        import pymht.models.pv as model
        import pymht.utils.kalman as kalman
        assert len(aisMeasurements) > 0
        aisPredictions = AisMessageList(scanTime)
        scanTimeString = datetime.datetime.fromtimestamp(scanTime).strftime("%H:%M:%S.%f")
        for measurement in aisMeasurements:
            aisTimeString = datetime.datetime.fromtimestamp(measurement.time).strftime("%H:%M:%S.%f")
            log.debug("Predicting AIS (" + str(measurement.mmsi) + ") from " + aisTimeString + " to " + scanTimeString)
            dT = scanTime - measurement.time
            assert dT >= 0
            state = measurement.state
            A = model.Phi(dT)
            Q = model.Q(dT)
            x_bar, P_bar = kalman.predict(A, Q, np.array(state, ndmin=2),
                                          np.array(measurement.covariance, ndmin=3))
            aisPredictions.measurements.append(
                AIS_prediction(model.C_RADAR.dot(x_bar[0]),
                               model.C_RADAR.dot(P_bar[0]).dot(model.C_RADAR.T), measurement.mmsi))
            log.debug(np.array_str(state) + "=>" + np.array_str(x_bar[0]))
            aisPredictions.aisMessages.append(measurement)
        assert len(aisPredictions.measurements) == len(aisMeasurements)
        return aisPredictions


class MeasurementList:

    def __init__(self, time, measurements=None):
        self.time = time
        self.measurements = measurements if measurements is not None else []

    def __str__(self):
        np.set_printoptions(precision=1, suppress=True)
        timeString = datetime.datetime.fromtimestamp(self.time).strftime("%H:%M:%S.%f")
        return ("Time: " + timeString +
                "\tMeasurements:\t" + ", ".join(
                    [str(measurement) for measurement in self.measurements]))

    def __eq__(self, other):
        if self.time != other.time:
            return False
        if not np.array_equal(self.measurements, other.measurements):
            return False
        return True

    __repr__ = __str__

    def plot(self, ax=plt.gca(), **kwargs):
        for measurementIndex, measurement in enumerate(self.measurements):
            Position(measurement).plot(ax, measurementIndex + 1, **kwargs)

    def filterUnused(self, unused_measurement_indices):
        measurements = self.measurements[np.where(unused_measurement_indices)]
        return MeasurementList(self.time, measurements)

    def getTimeString(self, timeFormat="%H:%M:%S"):
        return datetime.datetime.fromtimestamp(self.time).strftime(timeFormat)

    def getMeasurements(self):
        return self.measurements


class AisMessageList(list):

    def __init__(self, *args):
        list.__init__(self, *args)
        assert all([type(m) is AIS_message for m in self])
        mmsiList = [m.mmsi for m in self]
        deleteIndices = []
        duplicateMMSI = [item for item, count in collections.Counter(mmsiList).items() if count > 1]
        for mmsi in duplicateMMSI:
            indices = [(i, m.time) for i, m in enumerate(self) if m.mmsi == mmsi]
            indices.sort(key=lambda tup: tup[1])
            assert len(indices) > 1
            for tup in indices[:-1]:
                deleteIndices.append(tup[0])
        deleteIndices.sort(reverse=True)
        for i in deleteIndices:
            del self[i]

        mmsiList = [m.mmsi for m in self]
        mmsiSet = set(mmsiList)
        assert len(mmsiList) == len(mmsiSet)

    def filterUnused(self, usedMmsiSet):
        unusedAisMeasurements = [m for m in self
                                 if m.mmsi not in usedMmsiSet]
        return unusedAisMeasurements

    def plot(self, ax=plt.gca(), **kwargs):
        for measurement in self:
            measurement.plot(ax, **kwargs)


class ScanList(list):

    def __init__(self, *args):
        list.__init__(self, *args)
        assert all([type(m) is AIS_message for m in self])

    def append(self, item):
        if not isinstance(item, MeasurementList):
            raise TypeError('item is not of type' + str(type(MeasurementList)))
        super(ScanList, self).append(item)

    def plot(self, ax=plt.gca(), **kwargs):
        for m in self:
            m.plot(ax, **kwargs)

    def plotFast(self, ax=plt.gca(), **kwargs):
        for measurementList in self:
            measurementArray = np.array(measurementList.measurements, ndmin=2)
            assert measurementArray.ndim == 2
            assert measurementArray.shape[1] == 2
            ax.plot(measurementArray[:, 0], measurementArray[:, 1], '.', color='black', **kwargs)
