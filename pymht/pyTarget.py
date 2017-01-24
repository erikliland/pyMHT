try:
    raise ImportError
    import pymht.utils.ckalman as kalman
    print("Using cKalman")
except ImportError:
    print("Using pyKalman")
    import pymht.utils.pyKalman as kalman

# import pykalman.KalmanFilter as kf
# import pymht.utils.kalman as kf
import pymht.utils.helpFunctions as hpf
from pymht.utils.classDefinitions import Position, Velocity
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp


class Target():

    def __init__(self, **kwargs):
        # Process variables
        # super(Target,self).__init__()
        # self.exitFlag = False
        # self.queue = mp.Queue()
        # self.event = mp.Event()

        # Target variables
        time = kwargs.get("time")
        scanNumber = kwargs.get("scanNumber")
        filteredStateMean = kwargs.get("filteredStateMean")
        filteredStateCovariance = kwargs.get("filteredStateCovariance")
        if (	(time is None) or
                (scanNumber is None) or
                (filteredStateMean is None) or
                (filteredStateCovariance is None)):
            raise TypeError(
                "Target() need at least: time, scanNumber, state and covariance")

        # Track parameters
        self.time = time
        self.scanNumber = scanNumber
        self.parent = kwargs.get("parent")
        self.measurementNumber = kwargs.get("measurementNumber", 0)
        self.measurement = kwargs.get("measurement")
        self.cummulativeNLLR = kwargs.get("cummulativeNLLR", 0)
        self.trackHypotheses = []

        # Kalman filter variables
        Phi = kwargs.get("Phi")
        C = kwargs.get("C")
        Q = kwargs.get("Q")
        Gamma = kwargs.get("Gamma")
        R = kwargs.get("R")

        self.kalmanFilter = kalman.KalmanFilter(
            Phi,
            C,
            Q=Q,
            R=R,
            Gamma=Gamma,
            x_0=filteredStateMean,
            P_0=filteredStateCovariance
        )

    def __repr__(self):
        if self.kalmanFilter.x_bar is not None:
            np.set_printoptions(precision=4, suppress=True)
            predStateStr = " \tPredState: " + str(self.kalmanFilter.x_bar)
        else:
            predStateStr = ""

        if self.measurementNumber is not None:
            measStr = (" \tMeasurement(" + str(self.scanNumber)
                       + ":" + str(self.measurementNumber) + ")")
            if self.measurement is not None:
                measStr += ":" + str(self.measurement)
        else:
            measStr = ""

        if self.kalmanFilter.S is not None:
            lambda_, _ = np.linalg.eig(self.kalmanFilter.S)
            gateStr = (" \tGate size: (" + '{:5.2f}'.format(np.sqrt(lambda_[0]) * 2)
                       + "," + '{:5.2f}'.format(np.sqrt(lambda_[1]) * 2) + ")")
        else:
            gateStr = ""

        return ("Time: " + time.strftime("%H:%M:%S", time.gmtime(self.time))
                + "\t" + str(self.getPosition())
                + " \t" + str(self.getVelocity())
                        + " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR)
                        + measStr
                        + predStateStr
                        + gateStr
                )

    def __str__(self, **kwargs):
        level = kwargs.get("level", 0)
        hypIndex = kwargs.get("hypIndex", 0)
        targetIndex = kwargs.get("targetIndex", "?")

        if (level == 0) and not self.trackHypotheses:
            return repr(self)
        ret = ""
        if level == 0:
            ret += "T" + str(targetIndex) + ": " + repr(self) + "\n"
        else:
            ret += "   " + " " * min(level, 8) + "H" + \
                str(hypIndex) + ": " + repr(self) + "\n"
        for hypIndex, hyp in enumerate(self.trackHypotheses):
            hasNotZeroHyp = (self.trackHypotheses[0].measurementNumber != 0)
            ret += hyp.__str__(level=level + 1, hypIndex=hypIndex + int(hasNotZeroHyp))
        return ret

    def __sub__(self, other):
        return self.kalmanFilter.x_hat - other.kalmanFilter.x_hat

    def getPosition(self):
        pos = Position(self.kalmanFilter.x_hat[0:2])
        return pos

    def getVelocity(self):
        return Velocity(self.kalmanFilter.x_hat[2:4])

    def stepBack(self, stepsBack=1):
        if (stepsBack == 0) or (self.parent is None):
            return self
        return self.parent.stepBack(stepsBack - 1)

    def getRoot(self):
        return self.stepBack(float('inf'))

    def getNumOfNodes(self):
        if not self.trackHypotheses:
            return 1
        return 1 + sum([node.getNumOfNodes() for node in self.trackHypotheses])

    def depth(self, count=0):
        if len(self.trackHypotheses):
            return self.trackHypotheses[0].depth(count + 1)
        return count

    def predictMeasurement(self, scanTime):
        T = scanTime - self.time
        self.kalmanFilter.predict(T=T)
        self.kalmanFilter._precalculateMeasurementUpdate(T)

    def gateAndCreateNewHypotheses(self, measurementList, scanNumber, P_d, lambda_ex, eta2, **kwargs):
        tic = time.time()
        assert self.scanNumber == scanNumber - \
            1, "gateAndCreateNewMeasurement: inconsistent scan numbering"
        scanTime = measurementList.time
        associatedMeasurements = set()
        trackHypotheses = list()

        trackHypotheses.append(self.createZeroHypothesis(
            scanTime, scanNumber, P_d, **kwargs))

        measurementsResidual = measurementList.measurements - self.kalmanFilter.z_hat
        normalizedInnovationSquared = np.zeros(len(measurementList.measurements))
        for i, residual in enumerate(measurementsResidual):
            normalizedInnovationSquared[i] = residual.T.dot(
                self.kalmanFilter.S_inv).dot(residual)  # TODO: Vectorize this!
        #print("NIS", *normalizedInnovationSquared, sep = "\n", end = "\n\n")

        gatedMeasurements = normalizedInnovationSquared <= eta2

        for measurementIndex, insideGate in enumerate(gatedMeasurements):
            if insideGate:
                measurementResidual = measurementsResidual[measurementIndex]
                measurement = measurementList.measurements[measurementIndex]
                filtState, filtCov = self.kalmanFilter.filter(
                    y_tilde=measurementResidual, local=True)
                associatedMeasurements.add((scanNumber, measurementIndex + 1))
                trackHypotheses.append(
                    self.clone(
                        time=scanTime,
                        scanNumber=scanNumber,
                        measurementNumber=measurementIndex + 1,
                        measurement=measurement,
                        filteredStateMean=filtState,
                        filteredStateCovariance=filtCov,
                        cummulativeNLLR=self.calculateCNLLR(
                            P_d, measurementResidual, lambda_ex,
                            self.kalmanFilter.S, self.kalmanFilter.S_inv),
                        measurementResidual=measurementResidual,
                    )
                )
        toc = time.time() - tic
        return trackHypotheses, associatedMeasurements, tic

    def calculateCNLLR(self, P_d, measurementResidual, lambda_ex, resCov, invResCov):
        return (self.cummulativeNLLR +
                hpf.nllr(P_d, measurementResidual, lambda_ex, resCov, invResCov))

    def clone(self, **kwargs):
        time = kwargs.get("time")
        scanNumber = kwargs.get("scanNumber")
        filteredStateMean = kwargs.get("filteredStateMean")
        filteredStateCovariance = kwargs.get("filteredStateCovariance")
        cummulativeNLLR = kwargs.get("cummulativeNLLR")
        measurementNumber = kwargs.get("measurementNumber")
        measurement = kwargs.get("measurement")
        parent = kwargs.get("parent", self)
        Phi = kwargs.get("Phi", self.kalmanFilter.A)
        Q = kwargs.get("Q",  self.kalmanFilter.Q)
        Gamma = kwargs.get("Gamma", self.kalmanFilter.Gamma)
        C = kwargs.get("C",  self.kalmanFilter.C)
        R = kwargs.get("R",  self.kalmanFilter.R)

        return Target(
            time=time,
            scanNumber=scanNumber,
            filteredStateMean=filteredStateMean,
            filteredStateCovariance=filteredStateCovariance,
            parent=parent,
            measurementNumber=measurementNumber,
            measurement=measurement,
            cummulativeNLLR=cummulativeNLLR,
            Phi=Phi,
            Q=Q,
            Gamma=Gamma,
            C=C,
            R=R,
        )

    def measurementIsInsideErrorEllipse(self, measurement, eta2):
        measRes = measurement.position - self.predictedMeasurement
        return measRes.T.dot(self.invResidualCovariance).dot(measRes) <= eta2

    def createZeroHypothesis(self, time, scanNumber, P_d, **kwargs):
        return self.clone(time=time,
                          scanNumber=scanNumber,
                          measurementNumber=0,
                          filteredStateMean=self.kalmanFilter.x_bar,
                          filteredStateCovariance=self.kalmanFilter.P_bar,
                          cummulativeNLLR=self.cummulativeNLLR + hpf.nllr(P_d),
                          parent=kwargs.get("parent", self)
                          )

    def createZeroHypothesisDictionary(self, time, scanNumber, P_d, **kwargs):
        return self.__dict__

    def _pruneAllHypothesisExeptThis(self, keep):
        for hyp in self.trackHypotheses:
            if hyp != keep:
                self.trackHypotheses.remove(hyp)

    def getMeasurementSet(self, root=True):
        subSet = set()
        for hyp in self.trackHypotheses:
            subSet |= hyp.getMeasurementSet(False)
        if (self.measurementNumber == 0) or (root):
            return subSet
        else:
            return {(self.scanNumber, self.measurementNumber)} | subSet

    def processNewMeasurement(self, measurementList, measurementSet, scanNumber, P_d, lambda_ex, eta2):
        if not self.trackHypotheses:
            self.predictMeasurement(measurementList.time)
            trackHypotheses, newMeasurements, _ = self.gateAndCreateNewHypotheses(
                measurementList, scanNumber, P_d, lambda_ex, eta2)
            self.trackHypotheses = trackHypotheses
            measurementSet.update(newMeasurements)
        else:
            for hyp in self.trackHypotheses:
                hyp.processNewMeasurement(
                    measurementList, measurementSet, scanNumber, P_d, lambda_ex, eta2)

    def _selectBestHypothesis(self):
        def recSearchBestHypothesis(target, bestScore, bestHypothesis):
            if len(target.trackHypotheses) == 0:
                if target.cummulativeNLLR <= bestScore[0]:
                    bestScore[0] = target.cummulativeNLLR
                    bestHypothesis[0] = target
            else:
                for hyp in target.trackHypotheses:
                    recSearchBestHypothesis(hyp, bestScore, bestHypothesis)
        bestScore = [float('Inf')]
        bestHypothesis = np.empty(1, dtype=np.dtype(object))
        recSearchBestHypothesis(self, bestScore, bestHypothesis)
        return bestHypothesis

    def getLeafNodes(self):
        def recGetLeafNode(node, nodes):
            if not node.trackHypotheses:
                nodes.append(node)
            else:
                for hyp in node.trackHypotheses:
                    recGetLeafNode(hyp, nodes)
        nodes = []
        recGetLeafNode(self, nodes)
        return nodes

    def getLeafParents(self):
        leafNodes = self.getLeafNodes()
        parents = set()
        for node in leafNodes:
            parents.add(node.parent)
        return parents

    def pruneSimilarState(self, threshold):
        for hyp in self.trackHypotheses[1:]:
            deltaPos = np.linalg.norm(self.trackHypotheses[0] - hyp)
            if deltaPos <= threshold:
                self.trackHypotheses.pop(0)
                break

    def recursiveSubtractScore(self, score):
        if score == 0:
            return
        self.cummulativeNLLR -= score

        for hyp in self.trackHypotheses:
            hyp.recursiveSubtractScore(score)

    def _checkScanNumberIntegrety(self):
        assert type(
            self.scanNumber) is int, "self.scanNumber is not an integer %r" % self.scanNumber

        if self.parent is not None:
            assert type(
                self.parent.scanNumber) is int, "self.parent.scanNumber is not an integer %r" % self.parent.scanNumber
            assert self.parent.scanNumber == self.scanNumber - \
                1, "self.parent.scanNumber(%r) == self.scanNumber-1(%r)" % (
                    self.parent.scanNumber, self.scanNumber)

        for hyp in self.trackHypotheses:
            hyp._checkScanNumberIntegrety()

    def _checkReferenceIntegrety(self):
        def recCheckReferenceIntegrety(target):
            for hyp in target.trackHypotheses:
                assert hyp.parent == target, "Inconsistent parent <-> child reference: Measurement(" + str(target.scanNumber) + ":" + str(
                    target.measurementNumber) + ") <-> " + "Measurement(" + str(hyp.scanNumber) + ":" + str(hyp.measurementNumber) + ")"
                recCheckReferenceIntegrety(hyp)
        recCheckReferenceIntegrety(self.getRoot())

    def plotValidationRegion(self, eta2, stepsBack=0):
        if self.kalmanFilter.S is not None:
            self._plotCovarianceEllipse(eta2)
        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotValidationRegion(eta2, stepsBack - 1)

    def _plotCovarianceEllipse(self, eta2):
        from matplotlib.patches import Ellipse
        lambda_, _ = np.linalg.eig(self.kalmanFilter.S)
        ell = Ellipse(xy=(self.kalmanFilter.x_bar[0], self.kalmanFilter.x_bar[1]),
                      width=np.sqrt(lambda_[0]) * np.sqrt(eta2) * 2,
                      height=np.sqrt(lambda_[1]) * np.sqrt(eta2) * 2,
                      angle=np.rad2deg(np.arctan2(lambda_[1], lambda_[0])),
                      linewidth=2,
                      )
        ell.set_facecolor('none')
        ell.set_linestyle("dotted")
        ell.set_alpha(0.5)
        ax = plt.subplot(111)
        ax.add_artist(ell)

    def backtrackPosition(self, stepsBack=float('inf')):
        if self.parent is None:
            return [self.getPosition()]
        else:
            return self.parent.backtrackPosition(stepsBack) + [self.getPosition()]

    def plotTrack(self, stepsBack=float('inf'), **kwargs):
        colors = itertools.cycle(["r", "b", "g"])
        track = self.backtrackPosition(stepsBack)
        plt.plot([p.x() for p in track], [p.y() for p in track], **kwargs)

    def plotMeasurement(self, stepsBack=0, **kwargs):
        if self.measurement is not None:
            Position(self.measurement).plot(
                self.measurementNumber, self.scanNumber, **kwargs)
        elif kwargs.get("dummy", False):
            self.getPosition().plot(self.measurementNumber, self.scanNumber, **kwargs)

        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotMeasurement(stepsBack - 1, **kwargs)

    def plotVelocityArrow(self, stepsBack=1):
        if self.kalmanFilter.x_bar is not None:
            ax = plt.subplot(111)
            deltaPos = self.kalmanFilter.x_bar[0:2] - self.kalmanFilter.x_hat[0:2]
            ax.arrow(self.kalmanFilter.x_hat[0], self.kalmanFilter.x_hat[1], deltaPos[0], deltaPos[1],
                     head_width=0.1, head_length=0.1, fc="None", ec='k',
                     length_includes_head="true", linestyle="-", alpha=0.3, linewidth=1)
        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotVelocityArrow(stepsBack - 1)
