from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import copy
from pymht.utils.classDefinitions import MeasurementList, AIS_prediction
import pymht.utils.pyKalman as kalman
import pymht.models.pv as model


def _getBestTextPosition(normVelocity, **kwargs):
    DEBUG = kwargs.get('debug', False)
    compassHeading = np.arctan2(normVelocity[0], normVelocity[1]) * 180. / np.pi
    compassHeading = (compassHeading + 360.) % 360.
    assert compassHeading >= 0, str(compassHeading)
    assert compassHeading <= 360, str(compassHeading)
    quadrant = int(2 + (compassHeading - 90) // 90)
    assert quadrant >= 1, str(quadrant)
    assert quadrant <= 4, str(quadrant)
    assert type(quadrant) is int
    if DEBUG: print("Vector {0:} Heading {1:5.1f} Quadrant {2:}".format(normVelocity, compassHeading, quadrant))
    # return horizontal_alignment, vertical_alignment
    if quadrant == 1:
        return 'right', 'top'
    elif quadrant == 2:
        return 'right', 'bottom'
    elif quadrant == 3:
        return 'left', 'bottom'
    elif quadrant == 4:
        return 'left', 'top'
    else:
        print('_getBestTextPosition failed. Returning default')
        return 'center', 'center'


def binomial(n, k):
    return 1 if k == 0 else (0 if n == 0 else binomial(n - 1, k) + binomial(n - 1, k - 1))


def plotVelocityArrowFromNode(nodes, **kwargs):
    def recPlotVelocityArrowFromNode(node, stepsLeft):
        if node.predictedStateMean is not None:
            plotVelocityArrow(node)
        if stepsLeft > 0 and (node.parent is not None):
            recPlotVelocityArrowFromNode(node.parent, stepsLeft - 1)

    for node in nodes:
        recPlotVelocityArrowFromNode(node, kwargs.get("stepsBack", 1))


def plotRadarOutline(centerPosition, radarRange, **kwargs):
    from matplotlib.patches import Ellipse
    if kwargs.get("markCenter", True):
        plt.plot(centerPosition.x(), centerPosition.y(), "bo")
    ax = plt.subplot(111)
    circle = Ellipse((centerPosition.x(), centerPosition.y()), radarRange * 2, radarRange * 2,
                     edgecolor="black", linestyle="dotted", facecolor="none")
    ax.add_artist(circle)


def plotTrueTrack(simList, **kwargs):
    colors = kwargs.get("colors")
    newArgs = copy.copy(kwargs)
    if "colors" in newArgs:
        del newArgs["colors"]

    nScan = len(simList)
    nTargets = len(simList[0])
    posArray = np.zeros((nScan, nTargets, 2))
    for row, scan in enumerate(simList):
        posArray[row, :, :] = np.array([target.state[0:2] for target in scan])
    for col in range(nTargets):
        plt.plot(posArray[:, col, 0], posArray[:, col, 1], '.', alpha=0.7,
                 markeredgewidth=0.6, color=next(colors) if colors is not None else None, **newArgs)
    for col in range(nTargets):
        plt.plot(posArray[0, col, 0], posArray[0, col, 1], '.', color='black')


def printScanList(scanList):
    for index, measurement in enumerate(scanList):
        print("\tMeasurement ", index, ":\t", end='', sep='')
        measurement.print()


def printClusterList(clusterList):
    print("Clusters:")
    for clusterIndex, cluster in enumerate(clusterList):
        print("Cluster ", clusterIndex, " contains target(s):\t", cluster,
              sep="", end="\n")


def printHypothesesScore(targetList):
    def recPrint(target, targetIndex):
        if target.trackHypotheses is not None:
            for hyp in target.trackHypotheses:
                recPrint(hyp, targetIndex)

    for targetIndex, target in enumerate(targetList):
        print("\tTarget: ", targetIndex,
              "\tInit", target.initial.position,
              "\tPred", target.predictedPosition(),
              "\tMeas", target.measurement, sep="")


# def nllr(*args):
#     if len(args) == 1:
#         P_d = args[0]
#         if P_d == 1:
#             return -np.log(1e-6)
#         return -np.log(1 - P_d)
#     elif len(args) == 5:
#         P_d = args[0]
#         measurementResidual = args[1]
#         lambda_ex = args[2]
#         covariance = args[3]
#         invCovariance = args[4]
#         if (	(measurementResidual is not None) and
#                 (lambda_ex is not None) and
#                 (covariance is not None) and
#                 (invCovariance is not None)):
#             if lambda_ex == 0:
#                 print("RuntimeError('lambda_ex' can not be zero.)")
#                 lambda_ex += 1e-20
#             return (0.5 * (measurementResidual.T.dot(invCovariance).dot(measurementResidual))
#                     + np.log((lambda_ex * np.sqrt(np.linalg.det(2 * np.pi * covariance))) / P_d))
#     else:
#         raise ValueError("nllr() takes either 1 or 5 arguments (", len(args), ") given")


def backtrackMeasurementNumbers(selectedNodes, steps=None):
    def recBacktrackNodeMeasurements(node, measurementBacktrack, stepsLeft=None):
        if node.parent is not None:
            if stepsLeft is None:
                measurementBacktrack.append(node.measurementNumber)
                recBacktrackNodeMeasurements(node.parent, measurementBacktrack)
            elif stepsLeft > 0:
                measurementBacktrack.append(node.measurementNumber)
                recBacktrackNodeMeasurements(
                    node.parent, measurementBacktrack, stepsLeft - 1)

    measurementsBacktracks = []
    for node in selectedNodes:
        measurementNumberBacktrack = []
        recBacktrackNodeMeasurements(node, measurementNumberBacktrack, steps)
        measurementNumberBacktrack.reverse()
        measurementsBacktracks.append(measurementNumberBacktrack)
    return measurementsBacktracks


def backtrackNodePositions(selectedNodes, **kwargs):
    from classDefinitions import Position

    def recBacktrackNodePosition(node, measurementList):
        measurementList.append(Position(node.filteredStateMean[0:2]))
        if node.parent is not None:
            if node.parent.scanNumber != node.scanNumber - 1:
                raise ValueError("Inconsistent scanNumber-ing:",
                                 node.parent.scanNumber, "->", node.scanNumber)
            recBacktrackNodePosition(node.parent, measurementList)

    try:
        trackList = []
        for leafNode in selectedNodes:
            measurementList = []
            recBacktrackNodePosition(leafNode, measurementList)
            measurementList.reverse()
            trackList.append(measurementList)
        return trackList
    except ValueError as e:
        if kwargs.get("debug", False):
            print(e)
        raise


def writeTracksToFile(filename, trackList, time, **kwargs):
    f = open(filename, 'w')
    for targetTrack in trackList:
        s = ""
        for index, position in enumerate(targetTrack):
            s += str(position)
            s += ',' if index != len(targetTrack) - 1 else ''
        s += "\n"
        f.write(s)
    f.close()


def parseSolver(solverString):
    import pulp
    s = solverString.strip().lower()
    if s == "cplex":
        return pulp.CPLEX_CMD(None, 0, 1, 0, [])
    if s == "glpk":
        return pulp.GLPK_CMD(None, 0, 1, 0, [])
    if s == "cbc":
        return pulp.PULP_CBC_CMD()
    if s == "gurobi":
        return pulp.GUROBI_CMD(None, 0, 1, 0, [])
    if s == "pyglpk":
        return pulp.PYGLPK()
    print("Did not find solver", solverString, "\t Using default solver.")
    return None


def solverIsAvailable(solverString):
    s = solverString.strip().lower()
    if s == "cplex":
        return pulp.CPLEX_CMD().available() != False
    if s == "glpk":
        return pulp.GLPK_CMD().available() != False
    if s == "cbc":
        return pulp.PULP_CBC_CMD().available() != False
    if s == "gurobi":
        return pulp.GUROBI_CMD().available() != False
    return False


def predictAisMeasurements(scanTime, aisMeasurements):
    assert len(aisMeasurements) > 0
    aisPredictions = MeasurementList(scanTime)
    R = model.R()
    for measurement in aisMeasurements:
        dT = scanTime - measurement.time
        assert dT > 0
        state = measurement.state
        A = model.Phi(dT)
        Q = model.Q(dT)
        res = kalman.numpyPredict(A, model.C, Q, R, model.Gamma, np.array(state, ndmin=2), np.array(measurement.covariance, ndmin=3))
        x_bar, P_bar, _, _, _, _, _ = res
        aisPredictions.measurements.append(AIS_prediction(x_bar[0], P_bar[0], measurement.mmsi))
    return aisPredictions