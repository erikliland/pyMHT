import pymht.utils.helpFunctions as hpf
import pymht.utils.cFunctions as hpf2
from pymht.utils.classDefinitions import Position, Velocity
import numpy as np
import time
import copy
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp


class Target():

    def __init__(self, time, scanNumber, kalmanFilter, ** kwargs):
        self.time = time
        self.scanNumber = scanNumber
        self.kalmanFilter = kalmanFilter
        self.P_d = copy.copy(kwargs.get('P_d', 0.8))
        self.parent = kwargs.get("parent")
        self.measurementNumber = kwargs.get("measurementNumber", 0)
        self.measurement = kwargs.get("measurement")
        self.cummulativeNLLR = copy.copy(kwargs.get("cummulativeNLLR", 0))
        self.trackHypotheses = None

    def __repr__(self):
        if self.kalmanFilter.x_bar is not None:
            np.set_printoptions(precision=4, suppress=True)
            predStateStr = " \tPredState: " + str(self.kalmanFilter.x_bar)
        else:
            predStateStr = ""

        if self.measurementNumber is not None:
            measStr = (" \tMeasurement(" +
                       str(self.scanNumber) +
                       ":" +
                       str(self.measurementNumber) +
                       ")")
            if self.measurement is not None:
                measStr += ":" + str(self.measurement)
        else:
            measStr = ""

        if self.kalmanFilter.S is not None:
            lambda_, _ = np.linalg.eig(self.kalmanFilter.S)
            gateStr = (" \tGate size: (" +
                       '{:5.2f}'.format(np.sqrt(lambda_[0]) * 2) +
                       "," +
                       '{:5.2f}'.format(np.sqrt(lambda_[1]) * 2) +
                       ")")
        else:
            gateStr = ""

        return ("Time: " + time.strftime("%H:%M:%S", time.gmtime(self.time)) +
                "\t" + str(self.getPosition()) +
                " \t" + str(self.getVelocity()) +
                " \tcNLLR:" + '{: 06.4f}'.format(self.cummulativeNLLR) +
                measStr +
                predStateStr +
                gateStr
                )

    def __str__(self, **kwargs):
        level = kwargs.get("level", 0)
        hypIndex = kwargs.get("hypIndex", 0)
        targetIndex = kwargs.get("targetIndex", "?")

        if (level == 0) and self.trackHypotheses is None:
            return repr(self)
        ret = ""
        if level == 0:
            ret += "T" + str(targetIndex) + ": " + repr(self) + "\n"
        else:
            ret += "   " + " " * min(level, 8) + "H" + \
                str(hypIndex) + ": " + repr(self) + "\n"
        if self.trackHypotheses is not None:
            for hypIndex, hyp in enumerate(self.trackHypotheses):
                hasNotZeroHyp = (self.trackHypotheses[0].measurementNumber != 0)
                ret += hyp.__str__(level=level + 1,
                                   hypIndex=hypIndex + int(hasNotZeroHyp))
        return ret

    def __sub__(self, other):
        return self.kalmanFilter.x_hat - other.kalmanFilter.x_hat

    def getPosition(self):
        return Position(self.kalmanFilter.x_hat[0:2])

    def getVelocity(self):
        return Velocity(self.kalmanFilter.x_hat[2:4])

    def stepBack(self, stepsBack=1):
        if (stepsBack == 0) or (self.parent is None):
            return self
        return self.parent.stepBack(stepsBack - 1)

    def getRoot(self):
        return self.stepBack(float('inf'))

    def getNumOfNodes(self):
        if self.trackHypotheses is None:
            return 1
        return 1 + sum([node.getNumOfNodes() for node in self.trackHypotheses])

    def depth(self, count=0):
        return count if self.trackHypotheses is None else self.trackHypotheses[0].depth(count + 1)

    def predictMeasurement(self, **kwargs):
        # T = scanTime - self.time
        self.kalmanFilter.predict()
        self.kalmanFilter._precalculateMeasurementUpdate()

    def gateAndCreateNewHypotheses(self, measurementList, scanNumber, lambda_ex, eta2):
        # tic1 = time.process_time()
        assert self.scanNumber == scanNumber - 1, \
            "gateAndCreateNewMeasurement: inconsistent scan numbering"
        scanTime = measurementList.time
        measurementsResidual = measurementList.measurements - self.kalmanFilter.z_hat
        nis = self._normalizedInnovationSquared(measurementsResidual)
        gatedMeasurements = nis <= eta2
        nMeasurementInsideGate = np.sum(gatedMeasurements)
        associatedMeasurements = set()
        self.trackHypotheses = np.empty(shape=(nMeasurementInsideGate + 1,), dtype=object)
        self.trackHypotheses[0] = self.createZeroHypothesis(scanTime, scanNumber)
        i = 1
        # tic2 = time.process_time()
        for measurementIndex, insideGate in enumerate(gatedMeasurements):
            if insideGate:
                measurementResidual = measurementsResidual[measurementIndex]
                measurement = measurementList.measurements[measurementIndex]
                associatedMeasurements.add((scanNumber, measurementIndex + 1))
                self.trackHypotheses[i] = Target(
                    scanTime,
                    scanNumber,
                    self.kalmanFilter.filterAndCopy(measurementResidual),
                    measurementNumber=measurementIndex + 1,
                    measurement=measurement,
                    cummulativeNLLR=self.calculateCNLLR(measurementResidual, lambda_ex),
                    P_d=self.P_d,
                    parent=self
                )
                i += 1
        # toc2 = time.process_time() - tic2
        # toc1 = time.process_time() - tic1
        # if (toc1 > 0.002) or (toc2 > 0.002):
        #     print('({0:.1f}|{1:.1f}|{2:.1f})ms'.format(
        #         toc1 * 1000, toc2 * 1000, 0),
        #         nMeasurementInsideGate)
        return self.trackHypotheses, associatedMeasurements

    def _normalizedInnovationSquared(self, measurementsResidual):
        return np.sum(measurementsResidual.dot(self.kalmanFilter.S_inv) *
                      measurementsResidual, axis=1)

    def calculateCNLLR(self, measurementResidual, lambda_ex):
        return self.cummulativeNLLR + hpf.nllr(self.P_d,
                                               measurementResidual,
                                               lambda_ex,
                                               self.kalmanFilter.S,
                                               self.kalmanFilter.S_inv)

    def measurementIsInsideErrorEllipse(self, measurement, eta2):
        measRes = measurement.position - self.predictedMeasurement
        return measRes.T.dot(self.invResidualCovariance).dot(measRes) <= eta2

    def createZeroHypothesis(self, time, scanNumber):
        return Target(time,
                      scanNumber,
                      self.kalmanFilter.filterAndCopy(),
                      measurementNumber=0,
                      cummulativeNLLR=self.cummulativeNLLR - np.log(1 - self.P_d),
                      P_d=self.P_d,
                      parent=self
                      )

    def _pruneAllHypothesisExeptThis(self, keep):  # TODO: Vectorize this
        for hyp in self.trackHypotheses:
            if hyp != keep:
                index = np.where(self.trackHypotheses == hyp)
                self.trackHypotheses = np.delete(self.trackHypotheses, index)

    def getMeasurementSet(self, root=True):
        subSet = set()
        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                subSet |= hyp.getMeasurementSet(False)
        if (self.measurementNumber == 0) or (root):
            return subSet
        else:
            return {(self.scanNumber, self.measurementNumber)} | subSet

    def processNewMeasurementRec(self, measurementList, measurementSet, scanNumber, lambda_ex, eta2):
        if self.trackHypotheses is None:
            self.predictMeasurement()
            _, newMeasurements = self.gateAndCreateNewHypotheses(
                measurementList, scanNumber, lambda_ex, eta2)
            measurementSet.update(newMeasurements)
        else:
            for hyp in self.trackHypotheses:
                hyp.processNewMeasurementRec(
                    measurementList, measurementSet, scanNumber, lambda_ex, eta2)

    def _selectBestHypothesis(self):
        def recSearchBestHypothesis(target, bestScore, bestHypothesis):
            if target.trackHypotheses is None:
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
            if node.trackHypotheses is None:
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

        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                hyp.recursiveSubtractScore(score)

    def _checkScanNumberIntegrety(self):
        assert type(
            self.scanNumber) is int, "self.scanNumber is not an integer %r" % self.scanNumber

        if self.parent is not None:
            assert type(self.parent.scanNumber) is int, \
                "self.parent.scanNumber is not an integer %r" % self.parent.scanNumber
            assert self.parent.scanNumber == self.scanNumber - 1,\
                "self.parent.scanNumber(%r) == self.scanNumber-1(%r)" % (
                    self.parent.scanNumber, self.scanNumber)

        for hyp in self.trackHypotheses:
            hyp._checkScanNumberIntegrety()

    def _checkReferenceIntegrety(self):
        def recCheckReferenceIntegrety(target):
            if target.trackHypotheses is not None:
                for hyp in target.trackHypotheses:
                    assert hyp.parent == target, \
                        ("Inconsistent parent <-> child reference: Measurement(" +
                         str(target.scanNumber) + ":" + str(target.measurementNumber) +
                         ") <-> " + "Measurement(" + str(hyp.scanNumber) + ":" +
                         str(hyp.measurementNumber) + ")")
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
