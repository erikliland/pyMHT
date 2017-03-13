from pymht.utils.classDefinitions import Position, Velocity, MeasurementList
import pymht.models.pv as model
import pymht.utils.pyKalman as kalman
import pymht.utils.helpFunctions as hpf
import numpy as np
import time
import copy
import itertools
import queue
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp


class TargetManager(mp.Process):
    def __init__(self, q, idx):
        super(TargetManager, self).__init__()
        self.exit_flag = False
        self.queue = q
        self.idx = idx
        self.rootNode = None

    def exit_request(self):
        print(self.idx, "got exit flag")
        self.exit_flag = True
        print(self.idx, self.exit_flag)

    def parseInput(self, data):
        dataType = type(data)
        if dataType is MeasurementList:
            return self.processMeasurements

    def processMeasurements(self, measurementList):
        print("Processing measurements")
        pass

    def run(self):
        print("Starting run")
        while not self.exit_flag:
            try:
                data = self.queue.get()
                print(data)
                functionToCall = self.parseInput(data)
                functionToCall(*data)
            except queue.Empty:
                pass
        print("Exiting run and terminating")


class Target():
    def __init__(self, time, scanNumber, x_0, P_0, **kwargs):
        assert (scanNumber is None) or (scanNumber == int(scanNumber))
        assert x_0.ndim == 1
        assert P_0.ndim == 2, str(P_0.shape)
        assert x_0.shape[0] == P_0.shape[0] == P_0.shape[1]
        self.time = time
        self.scanNumber = scanNumber
        self.x_0 = x_0
        self.P_0 = P_0
        self.P_d = copy.copy(kwargs.get('P_d', 0.8))
        self.parent = kwargs.get("parent")
        self.measurementNumber = kwargs.get("measurementNumber", 0)
        self.measurement = kwargs.get("measurement")
        self.cumulativeNLLR = copy.copy(kwargs.get("cumulativeNLLR", 0))
        self.trackHypotheses = None
        self.mmsi = kwargs.get('mmsi')
        assert self.P_d >= 0
        assert self.P_d <= 1
        assert (type(self.parent) == type(self) or self.parent is None)
        assert (self.mmsi is None) or (self.mmsi > 1e8)

    def __repr__(self):
        if hasattr(self, 'kalmanFilter'):
            np.set_printoptions(precision=4, suppress=True)
            predStateStr = " \tPredState: " + str(self.kalmanFilter.x_bar)
        else:
            predStateStr = ""

        if (self.measurementNumber is not None) and (self.scanNumber is not None):
            measStr = (" \tMeasurement(" +
                       str(self.scanNumber) +
                       ":" +
                       str(self.measurementNumber) +
                       ")")
            if self.measurement is not None:
                measStr += ":" + str(self.measurement)
        else:
            measStr = ""

        if hasattr(self, 'kalmanFilter'):
            lambda_, _ = np.linalg.eig(self.kalmanFilter.S)
            gateStr = (" \tGate size: (" +
                       '{:5.2f}'.format(np.sqrt(lambda_[0]) * 2) +
                       "," +
                       '{:5.2f}'.format(np.sqrt(lambda_[1]) * 2) +
                       ")")
        else:
            gateStr = ""

        if self.cumulativeNLLR == 0.0:
            nllrStr = ""
        else:
            nllrStr = " \tcNLLR:" + '{: 06.4f}'.format(self.cumulativeNLLR)

        if self.mmsi is not None:
            mmsiString = " \tMMSI: " + str(self.mmsi)
        else:
            mmsiString = ""

        timeString = datetime.datetime.fromtimestamp(self.time).strftime("%H:%M:%S.%f")

        return ("Time: " + timeString +
                "\t" + str(self.getPosition()) +
                " \t" + str(self.getVelocity()) +
                nllrStr +
                measStr +
                predStateStr +
                gateStr +
                mmsiString
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
        return self.x_0 - other.x_0

    def getPosition(self):
        return Position(self.x_0[0:2])

    def getVelocity(self):
        return Velocity(self.x_0[2:4])

    def stepBack(self, stepsBack=1):
        if (stepsBack == 0) or (self.parent is None):
            return self
        return self.parent.stepBack(stepsBack - 1)

    def getInitial(self):
        return self.stepBack(float('inf'))

    def getNumOfNodes(self):
        if self.trackHypotheses is None:
            return 1
        return 1 + sum([node.getNumOfNodes() for node in self.trackHypotheses])

    def depth(self, count=0):
        return (count if self.trackHypotheses is None
                else self.trackHypotheses[0].depth(count + 1))

    def predictMeasurement(self, **kwargs):
        self.kalmanFilter.predict()
        self.kalmanFilter._precalculateMeasurementUpdate()

    def isOutsideRange(self, position, range):
        distance = np.linalg.norm(model.C_RADAR.dot(self.x_0) - position)
        return distance > range

    def gateAndCreateNewHypotheses(self, measurementList, scanNumber, lambda_ex, eta2, kfVars):
        assert self.scanNumber == scanNumber - 1, "inconsistent scan numbering"
        x_bar, P_bar, z_hat, S, S_inv, K, P_hat = kalman.precalc(
            *kfVars, self.x_0.reshape(1, 4), self.P_0.reshape(1, 4, 4))
        scanTime = measurementList.time
        z_list = measurementList.measurements
        z_tilde = z_list - z_hat
        nis = self._normalizedInnovationSquared(z_tilde, S_inv.reshape(2, 2))
        gatedMeasurements = nis <= eta2
        self.trackHypotheses = [
            self.createZeroHypothesis(scanTime, scanNumber, x_bar[0], P_bar[0])]
        newNodes = []
        usedMeasurementIndices = set()
        for measurementIndex, insideGate in enumerate(gatedMeasurements):
            if not insideGate: continue
            nllr = kalman.nllr(lambda_ex, self.P_d, S, nis[measurementIndex])[0]
            x_hat = kalman.numpyFilter(
                x_bar, K.reshape(4, 2), z_tilde[measurementIndex].reshape(1, 2)).reshape(4, )
            assert x_hat.shape == self.x_0.shape
            newNodes.append(Target(scanTime,
                                   scanNumber,
                                   x_hat,
                                   P_hat[0],
                                   measurementNumber=measurementIndex + 1,
                                   measurement=z_list[measurementIndex],
                                   cumulativeNLLR=self.cumulativeNLLR + nllr,
                                   P_d=self.P_d,
                                   parent=self
                                   )
                            )
            usedMeasurementIndices.add(measurementIndex)
        self.trackHypotheses.extend(newNodes)
        return usedMeasurementIndices

    def spawnNewNodes(self, scanTime, scanNumber, x_bar, P_bar, measurementsIndices,
                      measurements, states, covariance, nllrList, fusedAisData=None):
        assert scanTime > self.time
        assert self.scanNumber == scanNumber - 1, str(self.scanNumber) + "->" + str(scanNumber)
        assert x_bar.shape == (4,)
        assert P_bar.shape == (4, 4)
        assert all([state.shape == (4,) for state in states])
        assert covariance.shape == (4, 4)
        nNewRadarMeasurementsIndices = len(measurementsIndices)
        nNewStates = len(states)
        nNewScores = len(nllrList)
        assert nNewRadarMeasurementsIndices == nNewStates == nNewScores
        self.trackHypotheses = [self.createZeroHypothesis(
            scanTime, scanNumber, x_bar, P_bar)]

        self.trackHypotheses.extend(
            [Target(scanTime,
                    scanNumber,
                    x_0=states[i],
                    P_0=covariance,
                    measurementNumber=measurementsIndices[i] + 1,
                    measurement=measurements[measurementsIndices[i]],
                    cumulativeNLLR=self.cumulativeNLLR + nllrList[i],
                    P_d=self.P_d,
                    parent=self
                    ) for i in range(nNewStates)]
        )

        if fusedAisData is None: return
        (fusedStates,
         fusedCovariance,
         fusedMeasurementIndices,
         fusedNllr,
         fusedMMSI) = fusedAisData
        self.trackHypotheses.extend(
            [Target(scanTime,
                    scanNumber,
                    x_0=fusedStates[i],
                    P_0=fusedCovariance,
                    measurementNumber=fusedMeasurementIndices[i] + 1,
                    measurement=measurements[fusedMeasurementIndices[i]],
                    cumulativeNLLR=self.cumulativeNLLR + fusedNllr[i],
                    mmsi=fusedMMSI[i],
                    P_d=self.P_d,
                    parent=self)
             for i in range(len(fusedMeasurementIndices))])

    def _normalizedInnovationSquared(self, measurementsResidual, S_inv):
        return np.sum(measurementsResidual.dot(S_inv) *
                      measurementsResidual, axis=1)

    def calculateCNLLR(self, lambda_ex, measurementResidual, S, S_inv):
        P_d = self.P_d
        nis = measurementResidual.T.dot(S_inv).dot(measurementResidual)
        nllr = (0.5 * nis +
                np.log((lambda_ex * np.sqrt(np.linalg.det(2. * np.pi * S))) / P_d))
        return self.cumulativeNLLR + nllr

    def measurementIsInsideErrorEllipse(self, measurement, eta2):
        measRes = measurement.position - self.predictedMeasurement
        return measRes.T.dot(self.invResidualCovariance).dot(measRes) <= eta2

    def createZeroHypothesis(self, time, scanNumber, x_0, P_0):
        return Target(time,
                      scanNumber,
                      x_0,
                      P_0,
                      measurementNumber=0,
                      cumulativeNLLR=self.cumulativeNLLR - np.log(1 - self.P_d),
                      P_d=self.P_d,
                      parent=self
                      )

    def _pruneAllHypothesisExceptThis(self, keep, backtrack=False):
        indices = np.where(self.trackHypotheses != keep)
        self.trackHypotheses = np.delete(self.trackHypotheses, indices)

        if backtrack and self.parent is not None:
            self.parent._pruneAllHypothesisExceptThis(self, backtrack=backtrack)

    def _pruneEverythingExceptHistory(self):
        if self.parent is not None:
            self.parent._pruneAllHypothesisExceptThis(self, backtrack=True)

    def pruneDepth(self, stepsLeft):
        if stepsLeft <= 0:
            if self.parent is not None:
                self.parent._pruneAllHypothesisExceptThis(self, backtrack=True)
                self.recursiveSubtractScore(self.cumulativeNLLR)
                assert self.parent.scanNumber == self.scanNumber - 1, \
                    "nScanPruning2: from scanNumber" + str(self.parent.scanNumber) + "->" + str(self.scanNumber)
                return self
            else:
                return self
        elif self.parent is not None:
            return self.parent.pruneDepth(stepsLeft - 1)
        else:
            return self

    def pruneSimilarState(self, threshold):
        for hyp in self.trackHypotheses[1:]:
            deltaPos = np.linalg.norm(self.trackHypotheses[0] - hyp)
            if deltaPos <= threshold:
                self.trackHypotheses.pop(0)
                break

    def getMeasurementSet(self, root=True):
        subSet = set()
        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                subSet |= hyp.getMeasurementSet(False)
        if (self.measurementNumber == 0) or (root):
            return subSet
        else:
            return {(self.scanNumber, self.measurementNumber)} | subSet

    def processNewMeasurementRec(self, measurementList, usedMeasurementSet,
                                 scanNumber, lambda_ex, eta2, kfVars):
        if self.trackHypotheses is None:
            usedMeasurementIndices = self.gateAndCreateNewHypotheses(measurementList,
                                                                     scanNumber,
                                                                     lambda_ex,
                                                                     eta2,
                                                                     kfVars)
            usedMeasurementSet.update(usedMeasurementIndices)
        else:
            for hyp in self.trackHypotheses:
                hyp.processNewMeasurementRec(
                    measurementList, usedMeasurementSet, scanNumber, lambda_ex, eta2, kfVars)

    def _selectBestHypothesis(self):
        def recSearchBestHypothesis(target, bestScore, bestHypothesis):
            if target.trackHypotheses is None:
                if target.cumulativeNLLR <= bestScore[0]:
                    bestScore[0] = target.cumulativeNLLR
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

    def recursiveSubtractScore(self, score):
        if score == 0:
            return
        self.cumulativeNLLR -= score

        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                hyp.recursiveSubtractScore(score)

    def _checkScanNumberIntegrity(self):
        assert type(self.scanNumber) is int, \
            "self.scanNumber is not an integer %r" % self.scanNumber

        if self.parent is not None:
            assert type(self.parent.scanNumber) is int, \
                "self.parent.scanNumber is not an integer %r" % self.parent.scanNumber
            assert self.parent.scanNumber == self.scanNumber - 1, \
                "self.parent.scanNumber(%r) == self.scanNumber-1(%r)" % (
                    self.parent.scanNumber, self.scanNumber)
        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                hyp._checkScanNumberIntegrity()

    def _checkReferenceIntegrity(self):
        def recCheckReferenceIntegrety(target):
            if target.trackHypotheses is not None:
                for hyp in target.trackHypotheses:
                    assert hyp.parent == target, \
                        ("Inconsistent parent <-> child reference: Measurement(" +
                         str(target.scanNumber) + ":" + str(target.measurementNumber) +
                         ") <-> " + "Measurement(" + str(hyp.scanNumber) + ":" +
                         str(hyp.measurementNumber) + ")")
                    recCheckReferenceIntegrety(hyp)

        recCheckReferenceIntegrety(self.getInitial())

    def plotValidationRegion(self, eta2, stepsBack=0):
        if not hasattr(self, 'kalmanFilter'):
            raise NotImplementedError("plotValidationRegion is not functional in this version")
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

    def plotTrack(self,root = None, stepsBack=float('inf'), **kwargs):
        if kwargs.get('markInitial', False) and stepsBack == float('inf'):
            self.getInitial().markInitial()
        if kwargs.get('markRoot', False) and root is not None:
            root.markRoot()
        if kwargs.get('markEnd'):
            self.markEnd()
        # colors = itertools.cycle(["r", "b", "g"])
        track = self.backtrackPosition(stepsBack)
        plt.plot([p.x() for p in track], [p.y() for p in track], c=kwargs.get('c'))

    def plotMeasurement(self, stepsBack=0, **kwargs):
        if (self.measurement is not None) and kwargs.get('real', True):
            Position(self.measurement).plot(
                self.measurementNumber, self.scanNumber, **kwargs)
        if kwargs.get("dummy", False):
            self.getPosition().plot(self.measurementNumber, self.scanNumber, **kwargs)

        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotMeasurement(stepsBack - 1, **kwargs)

    def plotStates(self, stepsBack=0, **kwargs):
        if (self.measurementNumber == 0) and kwargs.get("dummy", False):
            Position(self.x_0).plot(self.measurementNumber,
                                    self.scanNumber,
                                    **kwargs)
        elif (self.measurementNumber > 0) and kwargs.get('real', True):
            Position(self.x_0).plot(self.measurementNumber,
                                    self.scanNumber,
                                    self.mmsi,
                                    **kwargs)
        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotStates(stepsBack - 1, **kwargs)

    def plotVelocityArrow(self, stepsBack=1):
        if self.kalmanFilter.x_bar is not None:
            ax = plt.subplot(111)
            deltaPos = self.kalmanFilter.x_bar[0:2] - self.kalmanFilter.x_hat[0:2]
            ax.arrow(self.kalmanFilter.x_hat[0],
                     self.kalmanFilter.x_hat[1],
                     deltaPos[0],
                     deltaPos[1],
                     head_width=0.1,
                     head_length=0.1,
                     fc="None", ec='k',
                     length_includes_head="true",
                     linestyle="-",
                     alpha=0.3,
                     linewidth=1)
        if (self.parent is not None) and (stepsBack > 0):
            self.parent.plotVelocityArrow(stepsBack - 1)

    def markInitial(self, **kwargs):
        plt.plot(self.x_0[0],
                 self.x_0[1],
                 "*",
                 markerfacecolor='None',
                 markeredgecolor='black')
        index = kwargs.get("index")
        if index is not None:
            ax = plt.subplot(111)
            normVelocity = (self.x_0[2:4] /
                            np.linalg.norm(self.x_0[2:4]))
            offset = kwargs.get('offset', np.zeros_like(normVelocity))
            position = self.x_0[0:2] - offset
            horizontalalignment, verticalalignment = hpf._getBestTextPosition(normVelocity)
            ax.text(position[0],
                    position[1],
                    "T" + str(index),
                    fontsize=8,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment)

    def markRoot(self):
        plt.plot(self.x_0[0],
                 self.x_0[1],
                 's',
                 markerfacecolor='None',
                 markeredgecolor = 'black')

    def markEnd(self):
        plt.plot(self.x_0[0],
                 self.x_0[1],
                 "h",
                 markerfacecolor='None',
                 markeredgecolor='black')

    def recPlotMeasurements(self, plottedMeasurements, **kwargs):
        if self.parent is not None:
            if self.measurementNumber == 0:
                self.plotMeasurement(**kwargs)
            else:
                if kwargs.get('real', True):
                    measurementID = (self.scanNumber, self.measurementNumber)
                    if measurementID not in plottedMeasurements:
                        self.plotMeasurement(**kwargs)
                        plottedMeasurements.add(measurementID)
        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                hyp.recPlotMeasurements(plottedMeasurements, **kwargs)

    def recPlotStates(self, **kwargs):
        if self.parent is not None:
            self.plotStates(**kwargs)
        if self.trackHypotheses is not None:
            for hyp in self.trackHypotheses:
                hyp.recPlotStates(**kwargs)


if __name__ == '__main__':
    pass
