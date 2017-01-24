"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Spring 2017
========================================================================================
"""
from __future__ import print_function

import pymht.utils.helpFunctions as hpf
try:
    from pymht.cTarget import Target
    print("Using cTarget")
except ImportError:
    from pymht.pyTarget import Target
    print("Using pyTarget")

import time
import signal
import os
import pulp
import itertools
import functools
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from scipy.sparse.csgraph import connected_components


def _setHightPriority():
    import psutil
    import platform
    p = psutil.Process(os.getpid())
    OS = platform.system()
    if (OS == "Darwin") or (OS == "Linux"):
        p.nice(5)
    elif OS == "Windows":
        p.nice(psutil.HIGH_PRIORITY_CLASS)


def initWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def addMeasurementToNode(measurementList, scanNumber,
                         P_d, lambda_ex, eta2, params):
    target, nodeIndex = params
    trackHypotheses, newMeasurements, nodeComputationTime = (
        target.gateAndCreateNewHypotheses(
            measurementList, scanNumber, P_d, lambda_ex, eta2))
    return nodeIndex, trackHypotheses, newMeasurements, nodeComputationTime


class Tracker():

    def __init__(self, Phi, C, Gamma, P_d, P0, R, Q, lambda_phi,
                 lambda_nu, eta2, N, solverStr, **kwargs):

        self.logTime = kwargs.get("logTime", False)
        self.debug = kwargs.get("debug", False)
        self.parallelize = kwargs.get("w", 1) > 1
        self.kwargs = kwargs

        if self.parallelize:
            self.nWorkers = max(kwargs.get("w") - 1, 1)
            self.workers = mp.Pool(self.nWorkers, initWorker)
        else:
            self.nWorkers = 0

        if self.debug:
            print("Using ", self.nWorkers + 1, " proceess(es) with ",
                  os.cpu_count(), " cores", sep="")

        # Tracker storage
        self.__targetList__ = []
        self.__scanHistory__ = []
        self.__associatedMeasurements__ = []
        self.__targetProcessList__ = []
        self.__trackNodes__ = np.empty(0, dtype=np.dtype(object))

        # Timing and logging
        if self.logTime:
            self.runtimeLog = {	'Total':	np.array([0.0, 0]),
                                'Process':	np.array([0.0, 0]),
                                'Cluster':	np.array([0.0, 0]),
                                'Optim':	np.array([0.0, 0]),
                                'Prune':	np.array([0.0, 0]),
                                }
            self.tic = np.zeros(5)
            self.toc = np.zeros(5)
            self.nOptimSolved = 0
            self.leafNodeTimeList = []
            self.createComputationTime = None

        # Tracker parameters
        self.P_d = P_d
        self.lambda_phi = lambda_phi
        self.lambda_nu = lambda_nu
        self.lambda_ex = lambda_phi + lambda_nu
        self.eta2 = eta2
        self.N = N
        self.solver = hpf.parseSolver(solverStr)
        self.pruneThreshold = kwargs.get("pruneThreshold")

        # State space model
        self.Phi = Phi
        self.b = np.zeros(4)
        self.C = C
        self.d = np.zeros(2)
        self.Gamma = Gamma
        self.P0 = P0
        self.R = R
        self.Q = Q

        if ((kwargs.get("realTime") is not None) and
                (kwargs.get("realTime") is True)):
            _setHightPriority()

        # Misc
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def __enter__(self):
        return self

    def __exit__(self, exeType, exeValue, traceback):
        if self.parallelize:
            self.workers.terminate()
            self.workers.join()

    def initiateTarget(self, newTarget):
        target = Target(time=newTarget.time,
                        scanNumber=len(self.__scanHistory__),
                        filteredStateMean=newTarget.state,
                        filteredStateCovariance=self.P0,
                        Phi=self.Phi,
                        Q=self.Q,
                        Gamma=self.Gamma,
                        C=self.C,
                        R=self.R
                        )
        self.__targetList__.append(target)
        self.__associatedMeasurements__.append(set())
        self.__trackNodes__ = np.append(self.__trackNodes__, target)

    def addMeasurementList(self, measurementList, **kwargs):
        if self.logTime:
            self.tic[0] = time.time()

        if kwargs.get("checkIntegrety", False):
            self._checkTrackerIntegrety()

        if self.logTime:
            self.tic[1] = time.time()
        self.__scanHistory__.append(measurementList)
        nMeas = len(measurementList.measurements)
        nTargets = len(self.__targetList__)
        scanNumber = len(self.__scanHistory__)

        if self.logTime:
            self.leafNodeTimeList = []
        for targetIndex, target in enumerate(self.__targetList__):
            if self.parallelize:
                targetStartDepth = target.depth()
                searchTic = time.time()
                leafNodes = target.getLeafNodes()
                seachToc = time.time() - searchTic
                predictTic = time.time()
                for node in leafNodes:
                    node.predictMeasurement(measurementList.time)
                predictToc = time.time() - predictTic
                createTic = time.time()
                floatChunkSize = len(leafNodes) / self.nWorkers
                workerIterations = 1
                chunkSize = int(np.ceil(floatChunkSize / workerIterations))
                self.createComputationTime = 0
                for nodeIndex, trackHypotheses, newMeasurements, nodeComputationTime in (
                        self.workers.imap_unordered(functools.partial(
                            addMeasurementToNode, measurementList, scanNumber, self.P_d, self.lambda_ex, self.eta2), zip(leafNodes, range(len(leafNodes))), chunkSize)):
                    leafNodes[nodeIndex].trackHypotheses = trackHypotheses
                    for hyp in leafNodes[nodeIndex].trackHypotheses:
                        hyp.parent = leafNodes[nodeIndex]
                    self.__associatedMeasurements__[targetIndex].update(newMeasurements)
                    self.createComputationTime += nodeComputationTime
                createToc = time.time() - createTic
                addToc = 0
                self.leafNodeTimeList.append([seachToc, predictToc, createToc, addToc])
                targetEndDepth = target.depth()
                assert targetEndDepth - \
                    1 == targetStartDepth, "Multithreaded 'processNewMeasurements' did not increase the target depth"
                target._checkReferenceIntegrety()
            else:
                target.processNewMeasurement(
                    measurementList, self.__associatedMeasurements__[targetIndex],
                    scanNumber, self.P_d, self.lambda_ex, self.eta2)

        if self.logTime:
            self.toc[1] = time.time() - self.tic[1]
        if self.parallelize and self.debug:
            print(*[round(e) for e in self.leafNodeTimeList], sep="ms\n", end="ms\n")

        if kwargs.get("printAssociation", False):
            print(*__associatedMeasurements__, sep="\n", end="\n\n")

        # --Cluster targets --
        if self.logTime:
            self.tic[2] = time.time()
        clusterList = self._findClustersFromSets()
        if self.logTime:
            self.toc[2] = time.time() - self.tic[2]
        if kwargs.get("printCluster", False):
            hpf.printClusterList(clusterList)

        # --Maximize global (cluster vise) likelihood--
        if self.logTime:
            self.tic[3] = time.time()
        self.nOptimSolved = 0
        for cluster in clusterList:
            if len(cluster) == 1:
                # self._pruneSmilarState(cluster, self.pruneThreshold)
                self.__trackNodes__[cluster] = self.__targetList__[
                    cluster[0]]._selectBestHypothesis()
            else:
                # self._pruneSmilarState(cluster, self.pruneThreshold/2)
                self.__trackNodes__[cluster] = self._solveOptimumAssociation(cluster)
                self.nOptimSolved += 1
        if self.logTime:
            self.toc[3] = time.time() - self.tic[3]

        if self.logTime:
            self.tic[4] = time.time()
        self._nScanPruning()
        if self.logTime:
            self.toc[4] = time.time() - self.tic[4]
        if self.logTime:
            self.toc[0] = time.time() - self.tic[0]

        if kwargs.get("checkIntegrety", False):
            self._checkTrackerIntegrety()

        if self.logTime:
            self.runtimeLog['Total'] += np.array([self.toc[0], 1])
            self.runtimeLog['Process'] += np.array([self.toc[1], 1])
            self.runtimeLog['Cluster'] += np.array([self.toc[2], 1])
            self.runtimeLog['Optim'] += np.array([self.toc[3], 1])
            self.runtimeLog['Prune'] += np.array([self.toc[4], 1])

        if kwargs.get("printInfo", False):
            print(	"Added scan number:", len(self.__scanHistory__),
                   " \tnMeas ", nMeas,
                   sep="")

        if kwargs.get("printTime", False):
            self.printTimeLog()
            if self.createComputationTime is not None:
                print("createComputationTime", self.createComputationTime)

        # Covariance consistance
        if "trueState" in kwargs:
            xTrue = kwargs.get("trueState")
            return self._compareTracksWithTruth(xTrue)

    def _compareTracksWithTruth(self, xTrue):
        return [	(target.filteredStateMean - xTrue[targetIndex].state).T.dot(
            np.linalg.inv(target.filteredStateCovariance)).dot(
            (target.filteredStateMean - xTrue[targetIndex].state))
            for targetIndex, target in enumerate(self.__trackNodes__)]

    def getRuntimeAverage(self, **kwargs):
        p = kwargs.get("precision", 3)
        if self.logTime:
            return {k: v[0] / v[1] for k, v in self.runtimeLog.items()}

    def _findClustersFromSets(self):
        self.superSet = set()
        for targetSet in self.__associatedMeasurements__:
            self.superSet |= targetSet
        nTargets = len(self.__associatedMeasurements__)
        nNodes = nTargets + len(self.superSet)
        adjacencyMatrix = np.zeros((nNodes, nNodes), dtype=bool)
        for targetIndex, targetSet in enumerate(self.__associatedMeasurements__):
            for measurementIndex, measurement in enumerate(self.superSet):
                adjacencyMatrix[targetIndex, measurementIndex +
                                nTargets] = (measurement in targetSet)
        (nClusters, labels) = connected_components(adjacencyMatrix)
        return [np.where(labels[:nTargets] == clusterIndex)[0] for clusterIndex in range(nClusters)]

    def getTrackNodes(self):
        return self.__trackNodes__

    def _solveOptimumAssociation(self, cluster):
        nHypInClusterArray = self._getHypInCluster(cluster)
        nRealMeasurementsInCluster = len(
            set.union(*[self.__associatedMeasurements__[i] for i in cluster]))
        problemSize = nRealMeasurementsInCluster * sum(nHypInClusterArray)

        t0 = time.time()
        (A1, measurementList) = self._createA1(
            nRealMeasurementsInCluster, sum(nHypInClusterArray), cluster)
        A2 = self._createA2(len(cluster), nHypInClusterArray)
        C = self._createC(cluster)
        t1 = time.time() - t0
        # print("matricesTime\t", round(t1,3))

        selectedHypotheses = self._solveBLP(A1, A2, C, len(cluster))
        selectedNodes = self._hypotheses2Nodes(selectedHypotheses, cluster)
        selectedNodesArray = np.array(selectedNodes)
        # print("Solving optimal association in cluster with targets",cluster,",   \t",
        # sum(nHypInClusterArray)," hypotheses and",nRealMeasurementsInCluster,"real measurements.",sep = " ")
        # print("nHypothesesInCluster",sum(nHypInClusterArray))
        # print("nRealMeasurementsInCluster", nRealMeasurementsInCluster)
        # print("nTargetsInCluster", len(cluster))
        # print("nHypInClusterArray",nHypInClusterArray)
        # print("c =", c)
        # print("A1", A1, sep = "\n")
        # print("size(A1)", A1.shape, "\t=>\t", nRealMeasurementsInCluster*sum(nHypInClusterArray))
        # print("A2", A2, sep = "\n")
        # print("measurementList",measurementList)
        # print("selectedHypotheses",selectedHypotheses)
        # print("selectedNodes",*selectedNodes, sep = "\n")
        # print("selectedNodesArray",*selectedNodesArray, sep = "\n")

        assert (len(selectedHypotheses) == len(cluster),
                "__solveOptimumAssociation did not find the correct number of hypotheses")
        assert (len(selectedNodes) == len(cluster),
                "__solveOptimumAssociation did not find the correct number of nodes")
        assert (len(selectedHypotheses) == len(set(selectedHypotheses)),
                "_solveOptimumAssociation selected two or more equal hyptheses")
        assert (len(selectedNodes) == len(set(selectedNodes)),
                "_solveOptimumAssociation found same node in more than one track in selectedNodes")
        assert (len(selectedNodesArray) == len(set(selectedNodesArray)),
                "_solveOptimumAssociation found same node in more than one track in selectedNodesArray")
        return selectedNodesArray

    def _getHypInCluster(self, cluster):
        def nLeafNodes(target):
            if not target.trackHypotheses:
                return 1
            else:
                return sum(nLeafNodes(hyp) for hyp in target.trackHypotheses)
        nHypInClusterArray = np.zeros(len(cluster), dtype=int)
        for i, targetIndex in enumerate(cluster):
            nHypInTarget = nLeafNodes(self.__targetList__[targetIndex])
            nHypInClusterArray[i] = nHypInTarget
        return nHypInClusterArray

    def _createA1(self, nRow, nCol, cluster):
        def recActiveMeasurement(target, A1, measurementList,
                                 activeMeasurements, hypothesisIndex):
            if len(target.trackHypotheses) == 0:
                # we are at a real measurement
                if ((target.measurementNumber != 0) and
                        (target.measurementNumber is not None)):
                    measurement = (target.scanNumber, target.measurementNumber)
                    try:
                        measurementIndex = measurementList.index(measurement)
                    except ValueError:
                        measurementList.append(measurement)
                        measurementIndex = len(measurementList) - 1
                    activeMeasurements[measurementIndex] = True
                    # print("Measurement list", measurementList)
                    # print("Measurement index", measurementIndex)
                    # print("HypInd = ", hypothesisIndex[0])
                    # print("Active measurement", activeMeasurements)
                A1[activeMeasurements, hypothesisIndex[0]] = True
                hypothesisIndex[0] += 1

            else:
                for hyp in target.trackHypotheses:
                    activeMeasurementsCpy = activeMeasurements.copy()
                    if ((hyp.measurementNumber != 0) and
                            (hyp.measurementNumber is not None)):
                        measurement = (hyp.scanNumber, hyp.measurementNumber)
                        try:
                            measurementIndex = measurementList.index(measurement)
                        except ValueError:
                            measurementList.append(measurement)
                            measurementIndex = len(measurementList) - 1
                        activeMeasurementsCpy[measurementIndex] = True
                    recActiveMeasurement(hyp, A1, measurementList,
                                         activeMeasurementsCpy, hypothesisIndex)
        A1 = np.zeros((nRow, nCol), dtype=bool)  # Numpy Array
        # A1 = sp.dok_matrix((nRow,nCol),dtype = bool) #pulp.sparse Matrix
        activeMeasurements = np.zeros(nRow, dtype=bool)
        measurementList = []
        hypothesisIndex = [0]
        # TODO:
        # http://stackoverflow.com/questions/15148496/python-passing-an-integer-by-reference
        for targetIndex in cluster:
            recActiveMeasurement(
                self.__targetList__[targetIndex], A1, measurementList, activeMeasurements, hypothesisIndex)
        return A1, measurementList

    def _createA2(self, nTargetsInCluster, nHypInClusterArray):
        A2 = np.zeros((nTargetsInCluster, sum(nHypInClusterArray)), dtype=bool)
        colOffset = 0
        for rowIndex, nHyp in enumerate(nHypInClusterArray):
            for colIndex in range(colOffset, colOffset + nHyp):
                A2[rowIndex, colIndex] = True
            colOffset += nHyp
        return A2

    def _createC(self, cluster):
        def getTargetScore(target, scoreArray):
            if len(target.trackHypotheses) == 0:
                scoreArray.append(target.cummulativeNLLR)
            else:
                for hyp in target.trackHypotheses:
                    getTargetScore(hyp, scoreArray)
        scoreArray = []
        for targetIndex in cluster:
            getTargetScore(self.__targetList__[targetIndex], scoreArray)
        return scoreArray

    def _hypotheses2Nodes(self, selectedHypotheses, cluster):
        def recDFS(target, selectedHypothesis, nodeList, counter):
            if len(target.trackHypotheses) == 0:
                if counter[0] in selectedHypotheses:
                    nodeList.append(target)
                counter[0] += 1
            else:
                for hyp in target.trackHypotheses:
                    recDFS(hyp, selectedHypotheses, nodeList, counter)
        nodeList = []
        counter = [0]
        for targetIndex in cluster:
            recDFS(self.__targetList__[targetIndex],
                   selectedHypotheses, nodeList, counter)
        return nodeList

    def _solveBLP(self, A1, A2, f, nHyp):
        (nMeas, nHyp) = A1.shape
        (nTargets, _) = A2.shape
        prob = pulp.LpProblem("Association problem", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
        c = pulp.LpVariable.dicts("c", range(nHyp))
        for i in range(len(f)):
            c[i] = f[i]
        prob += pulp.lpSum(c[i] * x[i] for i in range(nHyp))
        for row in range(nMeas):
            prob += pulp.lpSum([A1[row, col] * x[col]
                                for col in range(nHyp) if A1[row, col]]) <= 1
        for row in range(nTargets):
            prob += pulp.lpSum([A2[row, col] * x[col]
                                for col in range(nHyp) if A2[row, col]]) == 1
        tic = time.time()
        sol = prob.solve(self.solver)
        toc = time.time() - tic

        def getSelectedHyp1(p, threshold=0):
            hyp = [int(v.name[2:])
                   for v in p.variables() if abs(v.varValue - 1) <= threshold]
            hyp.sort()
            return hyp

        def getSelectedHyp2(p, threshold=0):
            hyp = [int(v[0][2:]) for v in p.variablesDict().items()
                   if abs(v[1].varValue - 1) <= threshold]
            hyp.sort()
            return hyp

        for threshold in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            selectedHypotheses = getSelectedHyp2(prob, threshold)
            if len(selectedHypotheses) == nHyp:
                break
        return selectedHypotheses

    def _nScanPruning(self):
        def recPruneNScan(node, targetIndex, stepsLeft):
            if stepsLeft <= 0:
                if node.parent is not None:
                    if self.__targetList__[targetIndex].scanNumber != node.scanNumber - 1:
                        raise ValueError("nScanPruning1: from scanNumber", self.__targetList__[
                                         targetIndex].scanNumber, "->", node.scanNumber)
                    changed = (self.__targetList__[targetIndex] != node)
                    assert self.__targetList__[targetIndex].depth() == node.parent.depth(
                    ), "nScanPrune: Inconsistent target.depth() and target.child.parent.depth()"

                    self.__targetList__[targetIndex] = node  # Checked
                    node.parent._pruneAllHypothesisExeptThis(node)  # Checked
                    node.recursiveSubtractScore(node.cummulativeNLLR)  # Checked
                    if node.parent.scanNumber != node.scanNumber - 1:
                        raise ValueError("nScanPruning2: from scanNumber",
                                         node.parent.scanNumber, "->", node.scanNumber)
                    return changed
                else:
                    return False
            elif node.parent is not None:
                return recPruneNScan(node.parent, targetIndex, stepsLeft - 1)
            else:
                return False

        for targetIndex, node in enumerate(self.__trackNodes__):
            changed = recPruneNScan(node, targetIndex, self.N)
            if changed:
                self.__associatedMeasurements__[targetIndex] = self.__targetList__[
                    targetIndex].getMeasurementSet()

    def _pruneSmilarState(self, cluster, threshold):
        for targetIndex in cluster:
            leafParents = self.__targetList__[targetIndex].getLeafParents()
            for node in leafParents:
                node.pruneSimilarState(threshold)

    def _checkTrackerIntegrety(self):
        assert len(self.__trackNodes__) == len(
            self.__targetList__), "There are not the same number trackNodes as targets"
        assert len(self.__targetList__) == len(set(self.__targetList__)
                                               ), "There are copies of targets in the target list"
        assert len(self.__trackNodes__) == len(set(self.__trackNodes__)
                                               ), "There are copies of track nodes in __trackNodes__"
        for target in self.__targetList__:
            target._checkScanNumberIntegrety()
            target._checkReferenceIntegrety()
        assert len({node.scanNumber for node in self.__trackNodes__}
                   ) == 1, "There are inconsistency in trackNodes scanNumber"

    def plotValidationRegionFromRoot(self, stepsBack=1):
        def recPlotValidationRegionFromTarget(target, eta2, stepsBack):
            if not target.trackHypotheses:
                target.plotValidationRegion(eta2, stepsBack)
            else:
                for hyp in target.trackHypotheses:
                    recPlotValidationRegionFromTarget(hyp, eta2, stepsBack)

        for target in self.__targetList__:
            recPlotValidationRegionFromTarget(target, self.eta2, stepsBack)

    def plotValidationRegionFromTracks(self, stepsBack=1):
        for node in self.__trackNodes__:
            node.plotValidationRegion(self.eta2, stepsBack)

    def plotHypothesesTrack(self, **kwargs):
        def recPlotHypothesesTrack(target, track=[], **kwargs):
            newTrack = track[:] + [target.getPosition()]
            if not target.trackHypotheses:
                plt.plot([p.x() for p in newTrack], [p.y()
                                                     for p in newTrack], "--", **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    recPlotHypothesesTrack(hyp,  newTrack, **kwargs)
        colors = kwargs.get("colors", self._getColorCycle())
        for target in self.__targetList__:
            recPlotHypothesesTrack(target, c=next(colors))

    def plotActiveTracks(self, **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for track in self.__trackNodes__:
            track.plotTrack(c=next(colors))

    def plotMeasurementsFromTracks(self, stepsBack=float('inf'), **kwargs):
        for node in self.__trackNodes__:
            node.plotMeasurement(stepsBack, **kwargs)

    def plotMeasurementsFromRoot(self, **kwargs):
        def recPlotMeasurements(target, plottedMeasurements, plotReal, plotDummy):
            if target.parent is not None:
                if target.measurementNumber == 0:
                    if plotDummy:
                        target.plotMeasurement(**kwargs)
                else:
                    if plotReal:
                        measurementID = (target.scanNumber, target.measurementNumber)
                        if measurementID not in plottedMeasurements:
                            target.plotMeasurement(**kwargs)
                            plottedMeasurements.add(measurementID)
            for hyp in target.trackHypotheses:
                recPlotMeasurements(hyp, plottedMeasurements, plotReal, plotDummy)

        if not (("real" in kwargs) or ("dummy" in kwargs)):
            return
        plottedMeasurements = set()
        for target in self.__targetList__:
            if kwargs.get("includeHistory", False):
                recPlotMeasurements(target.getRoot(), plottedMeasurements, kwargs.get(
                    "real", True), kwargs.get("dummy", True))
            else:
                recPlotMeasurements(target, plottedMeasurements, kwargs.get(
                    "real", True), kwargs.get("dummy", True))

    def plotScanIndex(self, index, **kwargs):
        self.__scanHistory__[index].plot(**kwargs)

    def plotLastScan(self, **kwargs):
        self.__scanHistory__[-1].plot(**kwargs)

    def plotAllScans(self, **kwargs):
        for scan in self.__scanHistory__:
            scan.plot(**kwargs)

    def plotVelocityArrowForTrack(self, stepsBack=1):
        for track in self.__trackNodes__:
            track.plotVelocityArrow(stepsBack)

    def plotInitialTargets(self, **kwargs):
        initialTargets = [target.getRoot() for target in self.__targetList__]
        for i, initialTarget in enumerate(initialTargets):
            index = kwargs.get("index", list(range(len(initialTargets))))
            if len(index) != len(initialTargets):
                raise ValueError(
                    "plotInitialTargets: Need equal number of targets and indecies")
            plt.plot(initialTarget.kalmanFilter.x_hat[0],
                     initialTarget.kalmanFilter.x_hat[1],
                     "k+")
            ax = plt.subplot(111)
            normVelocity = (
                initialTarget.kalmanFilter.x_hat[2:4] / np.linalg.norm(initialTarget.kalmanFilter.x_hat[2:4]))
            offset = 0.1 * normVelocity
            position = initialTarget.kalmanFilter.x_hat[0:2] - offset
            ax.text(position[0],
                    position[1],
                    "T" + str(index[i]),
                    fontsize=8,
                    horizontalalignment="center",
                    verticalalignment="center")

    def _getColorCycle(self):
        return itertools.cycle(self.colors)

    def printTargetList(self, **kwargs):
        print("TargetList:")
        for targetIndex, target in enumerate(self.__targetList__):
            if kwargs.get("backtrack", False):
                print(target.stepBack().__str__(targetIndex=targetIndex))
            else:
                print(target.__str__(targetIndex=targetIndex))
        print()

    def printTimeLog(self):
        if self.logTime:
            from termcolor import colored, cprint
            totalTime = self.toc[0]
            sumTime = sum(self.toc[1:4])
            deviation = ((totalTime - sumTime) / totalTime) > 0.015
            nNodes = sum([target.getNumOfNodes() for target in self.__targetList__])
            cprint(
                "Total time {0:6.0f}/{1:6.0f}ms".format(self.toc[0] * 1000, sumTime * 1000) +
                '  Process({0:3.0f}/{1:5.0f}) {2:6.1f}ms'.format(len(self.__scanHistory__[-1].measurements), nNodes, self.toc[1] * 1000) +
                '  Cluster {:5.1f}ms'.format(self.toc[2] * 1000) +
                '  Optim({0:g}) {1:5.1f}ms'.format(self.nOptimSolved, self.toc[3] * 1000) +
                '  Prune {:5.1f}ms'.format(self.toc[4] * 1000),
                'red' if deviation else None,
                attrs=['bold'] if 'period' in self.kwargs and totalTime > self.kwargs.get(
                    'period') * 0.5 else []
            )

        else:
            print("Logging not activated")
