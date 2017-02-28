import numpy as np
import matplotlib.pyplot as plt


class TempTarget:
    def __init__(self, state, time, P_d, disappearAfter = float('inf'), **kwargs):
        self.state = state
        self.time = time
        self.P_d = P_d
        self.disappearAfter = disappearAfter
        self.mmsi = kwargs.get('mmsi')


    def __str__(self):
        return ('Pos: ({0: 7.1f},{1: 7.1f})'.format(self.state[0], self.state[1]) + " " +
                'Vel: ({0: 5.1f},{1: 5.1f})'.format(self.state[2], self.state[3]) + " " +
                'Speed: {0:4.1f}m/s ({1:4.1f}knt)'.format(self.speed('m/s'), self.speed('knots')) + " " +
                'Pd: {:5.1f}%'.format(self.P_d*100.))

    def storeString(self):
        return ',{0:.2f},{1:.2f}'.format(*self.state[0:2])

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

    def plot(self, measurementNumber, scanNumber=None, **kwargs):
        if measurementNumber == 0:
            plt.plot(self.array[0], self.array[1],
                     color="black", fillstyle="none", marker="o")
        else:
            plt.plot(self.array[0], self.array[1], 'kx')
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


class MeasurementList:
    def __init__(self, Time, measurements=[]):
        self.time = Time
        self.measurements = measurements

    def __str__(self):
        from time import gmtime, strftime
        np.set_printoptions(precision=1, suppress=True)

        timeString = strftime("%H:%M:%S", gmtime(self.time))
        return ("Time: " + timeString +
                "\tMeasurements:\t" + "".join(
            [str(measurement) for measurement in self.measurements]))

    __repr__ = __str__

    def add(self, measurement):
        self.measurements.append(measurement)

    def plot(self, **kwargs):
        for measurementIndex, measurement in enumerate(self.measurements):
            Position(measurement).plot(measurementIndex + 1, **kwargs)

    def filter(self, unused_measurement_indices):
        # nMeas = unused_measurement_indices.shape[0]
        # mask = np.hstack((unused_measurement_indices.reshape(nMeas, 1),
        #                   unused_measurement_indices.reshape(nMeas, 1)))
        # measurements = np.ma.array(self.measurements, mask=np.logical_not(mask))
        measurements = self.measurements[np.where(unused_measurement_indices)]
        return MeasurementList(self.time, measurements)
