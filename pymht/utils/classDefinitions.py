import numpy as np
import matplotlib.pyplot as plt


class Position:

    def __init__(self, *args, **kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        if (x is not None) and (y is not None):
            self.position = np.array([x, y])
        elif len(args) == 1:
            self.position = np.array(args[0])
        elif len(args) == 2:
            self.position = np.array([args[0], args[1]])
        else:
            raise ValueError("Invalid arguments to Position")

    def __str__(self):
        return 'Pos: ({0: .2f},{1: .2f})'.format(self.position[0], self.position[1])

    def __repr__(self):
        return '({0:.3e},{1:.3e})'.format(self.position[0], self.position[1])

    def __add__(self, other):
        return Position(self.position + other.position)

    def __sub__(self, other):
        return Position(self.position - other.position)

    def __mul__(self, other):
        return Position(self.position * other.position)

    def __div__(self, other):
        return Position(self.position / other.position)

    def x(self):
        return self.position[0]

    def y(self):
        return self.position[1]

    def plot(self, measurementNumber, scanNumber=None, **kwargs):
        if measurementNumber == 0:
            plt.plot(self.position[0], self.position[1],
                     color="black", fillstyle="none", marker="o")
        else:
            plt.plot(self.position[0], self.position[1], 'kx')
        if (	(scanNumber is not None) and
                (measurementNumber is not None) and
                kwargs.get("labels", False)):
            ax = plt.subplot(111)
            ax.text(self.position[0], self.position[1], str(
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
        return 'Vel: ({: .2f},{: .2f})'.format(self.velocity[0], self.velocity[1])

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
