import numpy as np
import datetime
import matplotlib.pyplot as plt
import logging
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
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
        self.timeOfLastAisMessage = -float('inf')
        self.P_r = kwargs.get('P_r', 1.)

    def __str__(self):
        timeString = datetime.datetime.fromtimestamp(self.time).strftime("%H:%M:%S.%f")
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        return ('Time: ' + timeString + " " +
                'Pos: ({0: 7.1f},{1: 7.1f})'.format(self.state[0], self.state[1]) + " " +
                'Vel: ({0: 5.1f},{1: 5.1f})'.format(self.state[2], self.state[3]) + " " +
                'Speed: {0:4.1f}m/s ({1:4.1f}knt)'.format(self.speed('m/s'), self.speed('knots')) + " " +
                'Pd: {:3.0f}%'.format(self.P_d * 100.) + " " +
                mmsiString)

    __repr__ = __str__

    def __eq__(self, other):
        if not np.array_equal(self.state,other.state): return False
        if self.time != other.time: return False
        if self.P_d != other.P_d: return False
        if self.mmsi != other.mmsi: return False
        if self.sigma_Q != other.sigma_Q: return False
        if self.P_r != other.P_r: return False
        return True

    def inRange(self, p0, rRange):
        distance = np.linalg.norm(self.state[0:2] - p0)
        return distance <= rRange

    def storeString(self):
        return ',{0:.2f},{1:.2f}'.format(*self.state[0:2])

    def getXmlStateStrings(self, precision=2):
        return (str(round(self.state[0],precision)),
                str(round(self.state[1],precision)),
                str(round(self.state[2],precision)),
                str(round(self.state[3],precision)))

    def position(self):
        return Position(self.state[0], self.state[1])

    def velocity(self):
        return Velocity(self.state[2], self.state[3])

    def speed(self, unit='m/s'):
        speed_ms = np.linalg.norm(self.state[2:4])
        if unit == 'm/s':
            return speed_ms
        elif unit == 'knots':
            return speed_ms * 1.94384449
        else:
            raise ValueError("Unknown unit")

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

    def plot(self, measurementNumber=-1, scanNumber=None, mmsi=None, **kwargs):
        if mmsi is not None:
            marker = 'h' if kwargs.get('original',False) else 'D'
            plt.plot(self.array[0], self.array[1],
                     marker=marker, markerfacecolor='None',
                     markeredgewidth=kwargs.get('markeredgewidth',1)
                     )
        elif measurementNumber > 0:
            plt.plot(self.array[0], self.array[1], 'kx')
        elif measurementNumber == 0:
            plt.plot(self.array[0], self.array[1],
                     color="black", fillstyle="none", marker="o")
        else:
            raise ValueError("Not a valid measurement number")

        if ((scanNumber is not None) and
                (measurementNumber is not None) and
                kwargs.get("labels", False)):
            ax = plt.subplot(111)
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
        list.__init__(self,*args)

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
                                         attrib={typeTag:groundtruthTag,idTag:str(i)})
            statesElement = ET.SubElement(trackElement,
                                          statesTag)
            for j in range(nSamples):
                simTarget = self[j][i]
                inRange = simTarget.inRange(p0, radarRange)
                radarTime = ((simTarget.time-initialTime)%radarPeriod) == 0.
                if (not inRange) or (not radarTime):
                    continue
                stateElement = ET.SubElement(statesElement,
                                             stateTag,
                                             attrib={timeTag:str(simTarget.time),
                                                     pdTag:str(simTarget.P_d)})
                eastPos, northPos, eastVel, northVel = simTarget.getXmlStateStrings()
                positionElement = ET.SubElement(stateElement,positionTag)
                ET.SubElement(positionElement,northTag).text = northPos
                ET.SubElement(positionElement,eastTag).text = eastPos
                velocityElement = ET.SubElement(stateElement, velocityTag)
                ET.SubElement(velocityElement, northTag).text = northVel
                ET.SubElement(velocityElement, eastTag).text = eastVel
                if simTarget.mmsi is not None:
                    trackElement.attrib[mmsiTag] = str(simTarget.mmsi)
                    trackElement.attrib[aisclassTag] = str(simTarget.aisClass)
                    trackElement.attrib[prTag] = str(simTarget.P_r)
                statesElement.attrib[sigmaqTag] = str(simTarget.sigma_Q)
                trackElement.attrib[lengthTag] = str(j+1)



class AIS_message:
    def __init__(self, time, state, covariance, mmsi):
        self.time = time
        self.state = state
        self.covariance = covariance
        self.mmsi = mmsi

    def __str__(self):
        if self.time == int(self.time):
            timeFormat = "%H:%M:%S"
        else:
            timeFormat = "%H:%M:%S.%f"
        timeString = datetime.datetime.fromtimestamp(self.time).strftime(timeFormat)
        mmsiString = 'MMSI: ' + str(self.mmsi) if self.mmsi is not None else ""
        return ('Time: ' + timeString + " " +
                'State: ({0: 7.1f},{1: 7.1f},{2: 7.1f},{3: 7.1f})'.format(
                    self.state[0], self.state[1], self.state[2], self.state[3]) + " " +
                'Covariance diagonal: ' + np.array_str(np.diagonal(self.covariance),
                                                       precision=1,
                                                       suppress_small=True) + " " +
                mmsiString)


    def __eq__(self, other):
        if self.time != other.time: return False
        if not np.array_equal(self.state, other.state): return False
        if not np.array_equal(self.covariance, other.covariance): return False
        if self.mmsi != other.mmsi: return False
        return True

    __repr__ = __str__

    def plot(self, **kwargs):
        Position(self.state[0:2]).plot(mmsi=self.mmsi, original=True, **kwargs)


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


class AIS_messageList:
    def __init__(self,*args):
        self._list = list(*args)
        self._lastExtractedTime = None
        self._iterator = None
        self._nextAisMeasurements = None

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __iter__(self):
        return self._list.__iter__()

    def append(self,*args):
        self._list.append(*args)

    def pop(self,*args):
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
            if all((m.time < scanTime) for m in self._nextAisMeasurements):
                self._lastExtractedTime = scanTime
                res = self.predictAisMeasurements(scanTime, self._nextAisMeasurements)
                self._nextAisMeasurements = next(self._iterator, None)
                return res
        return None

    def predictAisMeasurements(self,scanTime, aisMeasurements):
        import pymht.models.pv as model
        import pymht.utils.kalman as kalman
        assert len(aisMeasurements) > 0
        aisPredictions = PredictionList(scanTime)
        scanTimeString = datetime.datetime.fromtimestamp(scanTime).strftime("%H:%M:%S.%f")
        for measurement in aisMeasurements:
            aisTimeString = datetime.datetime.fromtimestamp(measurement.time).strftime("%H:%M:%S.%f")
            log.debug("Predicting AIS ("+str(measurement.mmsi)+") from " + aisTimeString + " to " + scanTimeString)
            dT = scanTime - measurement.time
            assert dT > 0
            state = measurement.state
            A = model.Phi(dT)
            Q = model.Q(dT)
            x_bar, P_bar = kalman.predict(A, Q, model.Gamma, np.array(state, ndmin=2),
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
        if self.time != other.time: return False
        if not np.array_equal(self.measurements, other.measurements): return False
        return True

    __repr__ = __str__

    def plot(self, **kwargs):
        for measurementIndex, measurement in enumerate(self.measurements):
            Position(measurement).plot(measurementIndex + 1, **kwargs)

    def filterUnused(self, unused_measurement_indices):
        measurements = self.measurements[np.where(unused_measurement_indices)]
        return MeasurementList(self.time, measurements)

    def getTimeString(self, timeFormat="%H:%M:%S"):
        return datetime.datetime.fromtimestamp(self.time).strftime(timeFormat)

    def getMeasurements(self):
        return self.measurements


class PredictionList(MeasurementList):
    def __init__(self, time, predictions=None):
        MeasurementList.__init__(self, time, predictions)
        self.aisMessages = []

    def __str__(self):
        if self.time == int(self.time):
            timeFormat = "%H:%M:%S"
        else:
            timeFormat = "%H:%M:%S.%f"
        timeString = datetime.datetime.fromtimestamp(self.time).strftime(timeFormat)
        return ("Time: " + timeString +
                "\tMeasurements:\t" + ", ".join(
            [str(measurement) for measurement in self.measurements]))

    def plot(self, **kwargs):
        if kwargs.get('original', True):
            for message in self.aisMessages:
                message.plot(**kwargs)
        if kwargs.get('predicted', False):
            for measurement in self.measurements:
                Position(measurement.state[0:2]).plot(mmsi=measurement.mmsi, **kwargs)

    def getMeasurements(self):
        return np.array([m.state for m in self.measurements])
