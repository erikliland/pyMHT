"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Spring 2017
========================================================================================
"""
import pymht.utils.helpFunctions as hpf
import pymht.utils.pyKalman as kalman
import pymht.initiators.m_of_n as m_of_n
import time
import signal
import os
import logging
import copy
import sys
import itertools
import functools
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from scipy.sparse.csgraph import connected_components
from ortools.linear_solver import pywraplp


def _setHighPriority():
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


def addMeasurementToNode(measurementList, scanNumber, lambda_ex, eta2, params):
    target, nodeIndex = params
    trackHypotheses, newMeasurements = (
        target.gateAndCreateNewHypotheses(
            measurementList, scanNumber, lambda_ex, eta2))
    return nodeIndex, trackHypotheses, newMeasurements


def _getSelectedHyp1(p, threshold=0):
    hyp = [int(v.name[2:])
           for v in p.variables() if abs(v.varValue - 1) <= threshold]
    hyp.sort()
    return hyp


def _getSelectedHyp2(p, threshold=0):
    hyp = [int(v[0][2:]) for v in p.variablesDict().items()
           if abs(v[1].varValue - 1) <= threshold]
    hyp.sort()
    return hyp


class Tracker():
    def __init__(self, Phi, C, Gamma, P_d, P_0, R, Q, lambda_phi,
                 lambda_nu, eta2, N, p0, radarRange, solverStr, **kwargs):

        # ----------------------------------------------------------------------------
        # Instantiate logging object
        # ----------------------------------------------------------------------------
        self.log = logging.getLogger(__name__)
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setLevel(kwargs.get('logLevel', 'INFO'))
        self.log.setLevel(kwargs.get('logLevel', 'INFO'))
        log_format = '%(message)s'
        log_formatter = logging.Formatter(log_format)
        log_handler.setFormatter(log_formatter)
        self.log.handlers = []
        self.log.addHandler(log_handler)
        self.log.info('Running base main')

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

        # Target initiator
        maxSpeed = kwargs.get('maxSpeed', 20)
        M_required = kwargs.get('M', 2)
        N_checks = kwargs.get('N', 3)
        self.initiator = m_of_n.Initiator(M_required, N_checks, maxSpeed)

        # Tracker storage
        self.__targetList__ = []
        self.__targetWindowSize__ = []
        self.__scanHistory__ = []
        self.__associatedMeasurements__ = []
        self.__targetProcessList__ = []
        self.__trackNodes__ = np.empty(0, dtype=np.dtype(object))
        self.__terminatedTargets__ = []
        self.__clusterList__ = []

        # Radar parameters
        self.position = p0
        self.range = radarRange
        self.period = kwargs.get('period')
        self.fixedPeriod = 'period' in kwargs
        self.default_P_d = 0.8

        # Timing and logging
        self.runtimeLog = {'Total': np.array([0.0, 0]),
                           'Process': np.array([0.0, 0]),
                           'Cluster': np.array([0.0, 0]),
                           'Optim': np.array([0.0, 0]),
                           'N Prune': np.array([0.0, 0]),
                           'Term': np.array([0.0, 0]),
                           'Init': np.array([0.0, 0]),
                           'Window': np.array([0.0, 0]),
                           'ILP Prune': np.array([0.0, 0]),
                           }
        if self.logTime:
            self.tic = np.zeros(9)
            self.toc = np.zeros(9)
            self.nOptimSolved = 0
            self.leafNodeTimeList = []
            self.createComputationTime = None

        # Tracker parameters
        self.lambda_phi = lambda_phi
        self.lambda_nu = lambda_nu
        self.lambda_ex = lambda_phi + lambda_nu
        self.eta2 = eta2
        self.N_max = copy.copy(N)
        self.N = copy.copy(N)
        self.NLLR_UPPER_LIMIT = -(np.log(1 - 0.7)) * 7
        self.pruneThreshold = kwargs.get("pruneThreshold")

        # State space model
        self.A = Phi
        self.C = C
        self.Gamma = Gamma
        self.P_0 = P_0
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
        target = copy.copy(newTarget)
        target.scanNumber = len(self.__scanHistory__)
        target.P_d = self.default_P_d
        target.measurementNumber = None
        self.__targetList__.append(target)
        self.__associatedMeasurements__.append(set())
        self.__trackNodes__ = np.append(self.__trackNodes__, target)
        self.__targetWindowSize__.append(self.N)

    def addMeasurementList(self, measurementList, **kwargs):
        # 0 --Iterative procedure for tracking --
        self.tic[0] = time.time()

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 1 --Grow each track tree--
        if self.logTime:
            self.tic[1] = time.time()
        self.__scanHistory__.append(measurementList)
        nMeas = len(measurementList.measurements)
        measDim = self.C.shape[0]
        scanNumber = len(self.__scanHistory__)
        scanTime = measurementList.time
        nTargets = len(self.__targetList__)
        timeSinceLastScan = scanTime - self.__scanHistory__[-1].time
        if not self.fixedPeriod:
            self.period = timeSinceLastScan

        unused_measurement_indices = np.ones(nMeas, dtype=np.bool)

        if self.logTime:
            self.leafNodeTimeList = []

        targetProcessTimes = np.zeros(nTargets)
        for targetIndex, target in enumerate(self.__targetList__):
            if self.parallelize:
                targetStartDepth = target.depth()
                searchTic = time.time()
                leafNodes = target.getLeafNodes()
                searchToc = time.time() - searchTic
                predictTic = time.time()
                for node in leafNodes:
                    node.predictMeasurement()
                predictToc = time.time() - predictTic
                createTic = time.time()
                floatChunkSize = len(leafNodes) / self.nWorkers
                workerIterations = 1
                chunkSize = int(np.ceil(floatChunkSize / workerIterations))
                self.createComputationTime = 0
                for nodeIndex, trackHypotheses, newMeasurements in (
                        self.workers.imap_unordered(functools.partial(
                            addMeasurementToNode,
                            measurementList,
                            scanNumber,
                            self.lambda_ex,
                            self.eta2),
                            zip(leafNodes, range(len(leafNodes))), chunkSize)):
                    leafNodes[nodeIndex].trackHypotheses = trackHypotheses
                    for hyp in leafNodes[nodeIndex].trackHypotheses:
                        hyp.parent = leafNodes[nodeIndex]
                    self.__associatedMeasurements__[targetIndex].update(newMeasurements)
                createToc = time.time() - createTic
                addToc = 0
                self.leafNodeTimeList.append([searchToc, predictToc, createToc, addToc])
                targetEndDepth = target.depth()
                assert targetEndDepth - 1 == targetStartDepth, \
                    "'processNewMeasurements' did not increase the target depth"
                target._checkReferenceIntegrity()
            else:
                if kwargs.get('R', False):
                    usedMeasurementIndices = set()
                    target.processNewMeasurementRec(
                        measurementList,
                        usedMeasurementIndices,
                        scanNumber,
                        self.lambda_ex,
                        self.eta2,
                        (self.A, self.C, self.Q, self.R, self.Gamma)
                    )
                    for i in usedMeasurementIndices:
                        self.__associatedMeasurements__[targetIndex].update({(scanNumber, i + 1)})
                        unused_measurement_indices[i] = False

                else:
                    tic = time.time()
                    targetNodes = target.getLeafNodes()
                    nNodes = len(targetNodes)
                    newNodesData = self._processTargetNodes(targetNodes, measurementList)
                    x_bar_list, P_bar_list, gated_x_hat_list, P_hat_list, gatedIndicesList, nllrList = newNodesData
                    gatedMeasurementsList = [np.array(measurementList.measurements[gatedIndices])
                                             for gatedIndices in gatedIndicesList]
                    assert len(gatedMeasurementsList) == nNodes
                    assert all([m.shape[1] == measDim for m in gatedMeasurementsList])

                    for gated_index in gatedIndicesList:
                        unused_measurement_indices[gated_index] = False

                    for i, node in enumerate(targetNodes):
                        node.spawnNewNodes(scanTime,
                                           scanNumber,
                                           x_bar_list[i],
                                           P_bar_list[i],
                                           gatedIndicesList[i],
                                           gatedMeasurementsList[i],
                                           gated_x_hat_list[i],
                                           P_hat_list[i],
                                           nllrList[i])

                    for i in range(nNodes):
                        for measurementIndex in gatedIndicesList[i]:
                            self.__associatedMeasurements__[targetIndex].update(
                                {(scanNumber, measurementIndex + 1)})
                    targetProcessTimes[targetIndex] = time.time() - tic

        if self.logTime:
            self.toc[1] = time.time() - self.tic[1]
        if self.parallelize and self.debug:
            print(*[round(e) for e in self.leafNodeTimeList], sep="ms\n", end="ms\n")
        if kwargs.get("printAssociation", False):
            print(*self.__associatedMeasurements__, sep="\n", end="\n\n")

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 2 --Cluster targets --
        if self.logTime:
            self.tic[2] = time.time()
        self.__clusterList__ = self._findClustersFromSets()
        if self.logTime:
            self.toc[2] = time.time() - self.tic[2]
        if kwargs.get("printCluster", False):
            hpf.printClusterList(self.__clusterList__)

        # 3 --Maximize global (cluster vise) likelihood--
        if self.logTime:
            self.tic[3] = time.time()
        self.nOptimSolved = 0
        for cluster in self.__clusterList__:
            if len(cluster) == 1:
                if self.kwargs.get('pruneSimilar', False):
                    self._pruneSimilarState(cluster, self.pruneThreshold)
                self.__trackNodes__[cluster] = self.__targetList__[
                    cluster[0]]._selectBestHypothesis()
            else:
                self.__trackNodes__[cluster] = self._solveOptimumAssociation(cluster)
                self.nOptimSolved += 1
        if self.logTime:
            self.toc[3] = time.time() - self.tic[3]

        # 4 -- ILP Pruning
        if self.logTime:
            self.tic[4] = time.time()

        if self.logTime:
            self.toc[4] = time.time() - self.tic[4]

        # 5 -- Dynamic window size
        if self.logTime:
            self.tic[5] = time.time()

        for targetIndex, target in enumerate(self.__targetList__):
            targetProcessTime = targetProcessTimes[targetIndex]
            targetSize = target.getNumOfNodes()
            tooSlow = targetProcessTime > 100e-3
            tooLarge = targetSize > 3000
            if tooSlow or tooLarge:
                target = self.__targetList__[targetIndex]
                targetDepth = target.depth()
                assert targetDepth <= self.__targetWindowSize__[targetIndex] + 1
                infoString = "\tTarget {:2} ".format(targetIndex + 1)
                if tooSlow:
                    infoString += "Too slow {:.1f}ms. ".format(targetProcessTime * 1000)
                if tooLarge:
                    infoString += "To large {:}. ".format(targetSize)
                oldN = self.__targetWindowSize__[targetIndex]
                self.__targetWindowSize__[targetIndex] -= 1
                newN = self.__targetWindowSize__[targetIndex]
                infoString += "Reducing window from {0:} to {1:}".format(oldN, newN)
                self.log.debug(infoString)

        tempTotalTime = time.time() - self.tic[0]
        if tempTotalTime > (self.period * 0.8):
            self.N = max(1, self.N - 1)
            self.log.debug(
                '\tIteration took to long time ({0:.1f}ms), reducing window size roof from {1:} to  {2:}'.format(
                    tempTotalTime * 1000, self.N + 1, self.N))
            self.__targetWindowSize__ = [min(e, self.N) for e in self.__targetWindowSize__]

        if self.logTime:
            self.toc[5] = time.time() - self.tic[5]

        # 5 --Prune sliding window --
        if self.logTime:
            self.tic[6] = time.time()
        self._nScanPruning()
        if self.logTime:
            self.toc[6] = time.time() - self.tic[6]

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 6 -- Pick out dead tracks (terminate)
        if self.logTime:
            self.tic[7] = time.time()
        deadTracks = []
        for trackIndex, trackNode in enumerate(self.__trackNodes__):
            # Check outside range
            if trackNode.isOutsideRange(self.position.array, self.range):
                deadTracks.append(trackIndex)
                self.log.debug("\tTerminating track {:} since it is out of range".format(trackIndex))

            # Check if track is to insecure
            elif trackNode.cumulativeNLLR > self.NLLR_UPPER_LIMIT:
                deadTracks.append(trackIndex)
                self.log.debug("\tTerminating track {:} since its cost is above the threshold".format(trackIndex))

        self._terminateTracks(deadTracks)
        if self.logTime:
            self.toc[7] = time.time() - self.tic[7]

        # 7 -- Initiate new tracks
        if self.logTime:
            self.tic[8] = time.time()
        unused_measurements = measurementList.filter(unused_measurement_indices)
        new_initial_targets = self.initiator.processMeasurements(unused_measurements)
        for initial_target in new_initial_targets:
            self.log.debug("\tNew target({}): ".format(len(self.__targetList__) + 1) + str(initial_target))
            self.initiateTarget(initial_target)
        if self.logTime:
            self.toc[8] = time.time() - self.tic[8]

        self.toc[0] = time.time() - self.tic[0]

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        if self.logTime:
            self.runtimeLog['Total'] += np.array([self.toc[0], 1])
            self.runtimeLog['Process'] += np.array([self.toc[1], 1])
            self.runtimeLog['Cluster'] += np.array([self.toc[2], 1])
            self.runtimeLog['Optim'] += np.array([self.toc[3], 1])
            self.runtimeLog['ILP Prune'] += np.array([self.toc[4], 1])
            self.runtimeLog['Window'] += np.array([self.toc[5], 1])
            self.runtimeLog['N Prune'] += np.array([self.toc[6], 1])
            self.runtimeLog['Term'] += np.array([self.toc[7], 1])
            self.runtimeLog['Init'] += np.array([self.toc[8], 1])

        if kwargs.get("printInfo", False):
            print("Added scan number:", len(self.__scanHistory__),
                  " \tnMeas ", nMeas,
                  sep="")

        if kwargs.get("printTime", False):
            if self.logTime:
                self.printTimeLog()

        # Covariance consistence
        if "trueState" in kwargs:
            xTrue = kwargs.get("trueState")
            return self._compareTracksWithTruth(xTrue)

    def _terminateTracks(self, deadTracks):
        deadTracks.sort(reverse=True)
        for trackIndex in deadTracks:
            nTargetPre = len(self.__targetList__)
            nTracksPre = self.__trackNodes__.shape[0]
            nAssociationsPre = len(self.__associatedMeasurements__)
            targetListTypePre = type(self.__targetList__)
            trackListTypePre = type(self.__trackNodes__)
            associationTypePre = type(self.__associatedMeasurements__)
            self.__terminatedTargets__.append(copy.copy(self.__trackNodes__[trackIndex]))
            del self.__targetList__[trackIndex]
            del self.__targetWindowSize__[trackIndex]
            self.__trackNodes__ = np.delete(self.__trackNodes__, trackIndex)
            del self.__associatedMeasurements__[trackIndex]
            self.__terminatedTargets__[-1]._pruneEverythingExceptHistory()
            nTargetsPost = len(self.__targetList__)
            nTracksPost = self.__trackNodes__.shape[0]
            nAssociationsPost = len(self.__associatedMeasurements__)
            targetListTypePost = type(self.__targetList__)
            trackListTypePost = type(self.__trackNodes__)
            associationTypePost = type(self.__associatedMeasurements__)
            assert nTargetsPost == nTargetPre - 1
            assert nTracksPost == nTracksPre - 1, str(nTracksPre) + '=>' + str(nTracksPost)
            assert nAssociationsPost == nAssociationsPre - 1
            assert targetListTypePost == targetListTypePre
            assert trackListTypePost == trackListTypePre
            assert associationTypePost == associationTypePre

    def _processTargetNodes(self, targetNodes, measurementList):
        nMeas = len(measurementList.measurements)
        measDim = self.C.shape[0]
        nNodes = len(targetNodes)

        x_bar_list, P_bar_list, z_hat_list, S_list, S_inv_list, K_list, P_hat_list = self._predictPrecalcBulk(
            targetNodes)

        z_list = measurementList.measurements
        assert z_list.shape[1] == measDim

        z_tilde_list = kalman.z_tilde(z_list, z_hat_list, nNodes, measDim)
        assert z_tilde_list.shape == (nNodes, nMeas, measDim)

        nis = kalman.normalizedInnovationSquared(z_tilde_list, S_inv_list)
        assert nis.shape == (nNodes, nMeas,)

        gatedFilter = nis <= self.eta2
        assert gatedFilter.shape == (nNodes, nMeas)

        gatedIndicesList = [np.nonzero(gatedFilter[row])[0]
                            for row in range(nNodes)]

        assert len(gatedIndicesList) == nNodes

        gated_z_tilde_list = [z_tilde_list[i, gatedIndicesList[i]]
                              for i in range(nNodes)]
        assert len(gated_z_tilde_list) == nNodes

        gated_x_hat_list = [kalman.numpyFilter(
            x_bar_list[i], K_list[i], gated_z_tilde_list[i])
                            for i in range(nNodes)]
        assert len(gated_x_hat_list) == nNodes

        nllrList = [kalman.nllr(self.lambda_ex,
                                targetNodes[i].P_d,
                                S_list[i],
                                nis[i, gatedFilter[i]])
                    for i in range(nNodes)]
        assert len(nllrList) == nNodes

        # Return (x_bar, P_bar, x_hat_list, P_hat_list, measurementIndices, NLLR_list)
        newNodesData = (x_bar_list, P_bar_list, gated_x_hat_list, P_hat_list, gatedIndicesList, nllrList)
        return newNodesData

    def _predictPrecalcBulk(self, targetNodes):
        nNodes = len(targetNodes)
        measDim, nStates = self.C.shape
        x_0_list = np.array([target.x_0 for target in targetNodes],
                            ndmin=2)
        P_0_list = np.array([target.P_0 for target in targetNodes],
                            ndmin=3)
        assert x_0_list.shape == (nNodes, nStates)
        assert P_0_list.shape == (nNodes, nStates, nStates)

        x_bar_list, P_bar_list, z_hat_list, S_list, S_inv_list, K_list, P_hat_list = kalman.numpyPredict(
            self.A, self.C, self.Q, self.R, self.Gamma, x_0_list, P_0_list)

        assert x_bar_list.shape == x_0_list.shape
        assert x_bar_list.ndim == 2
        assert P_bar_list.shape == P_0_list.shape
        assert S_list.shape == (nNodes, measDim, measDim)
        assert S_inv_list.shape == (nNodes, measDim, measDim)
        assert K_list.shape == (nNodes, nStates, measDim)
        assert P_hat_list.shape == P_bar_list.shape
        assert z_hat_list.shape == (nNodes, measDim)

        return x_bar_list, P_bar_list, z_hat_list, S_list, S_inv_list, K_list, P_hat_list

    def _compareTracksWithTruth(self, xTrue):
        return [(target.filteredStateMean - xTrue[targetIndex].state).T.dot(
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
        return [np.where(labels[:nTargets] == clusterIndex)[0]
                for clusterIndex in range(nClusters)]

    def getTrackNodes(self):
        return self.__trackNodes__

    def _solveOptimumAssociation(self, cluster):
        nHypInClusterArray = self._getHypInCluster(cluster)
        nRealMeasurementsInCluster = len(
            set.union(*[self.__associatedMeasurements__[i] for i in cluster]))
        # problemSize = nRealMeasurementsInCluster * sum(nHypInClusterArray)

        # t0 = time.time()
        (A1, measurementList) = self._createA1(
            nRealMeasurementsInCluster, sum(nHypInClusterArray), cluster)
        A2 = self._createA2(len(cluster), nHypInClusterArray)
        C = self._createC(cluster)
        # t1 = time.time() - t0
        # print("matricesTime\t", round(t1,3))

        selectedHypotheses = self._solveBLP_OR_TOOLS(A1, A2, C, len(cluster))
        # selectedHypotheses = self._solveBLP_PULP(A1, A2, C, len(cluster))
        # assert selectedHypotheses0 == selectedHypotheses
        selectedNodes = self._hypotheses2Nodes(selectedHypotheses, cluster)
        selectedNodesArray = np.array(selectedNodes)
        self.log.debug("Solving optimal association in cluster with targets" + str(cluster) + ",   \t" +
                       str(sum(nHypInClusterArray)) + " hypotheses and " +
                       str(nRealMeasurementsInCluster) + " real measurements.")
        # print("nHypothesesInCluster",sum(nHypInClusterArray))
        # print("nRealMeasurementsInCluster", nRealMeasurementsInCluster)
        # print("nTargetsInCluster", len(cluster))
        # print("nHypInClusterArray",nHypInClusterArray)
        # print("c =", c)
        # print("A1", A1, sep = "\n")
        # print("size(A1)", A1.shape, "\t=>\t",
        #       nRealMeasurementsInCluster * sum(nHypInClusterArray))
        # print("A2", A2, sep = "\n")
        # print("measurementList",measurementList)
        # print("selectedHypotheses",selectedHypotheses)
        # print("selectedNodes",*selectedNodes, sep = "\n")
        # print("selectedNodesArray",*selectedNodesArray, sep = "\n")

        assert len(selectedHypotheses) == len(cluster), \
            "__solveOptimumAssociation did not find the correct number of hypotheses"
        assert len(selectedNodes) == len(cluster), \
            "did not find the correct number of nodes"
        assert len(selectedHypotheses) == len(set(selectedHypotheses)), \
            "selected two or more equal hyptheses"
        assert len(selectedNodes) == len(set(selectedNodes)), \
            "found same node in more than one track in selectedNodes"
        assert len(selectedNodesArray) == len(set(selectedNodesArray)), \
            "found same node in more than one track in selectedNodesArray"
        return selectedNodesArray

    def _getHypInCluster(self, cluster):
        def nLeafNodes(target):
            if target.trackHypotheses is None:
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
            if target.trackHypotheses is None:
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
        activeMeasurements = np.zeros(nRow, dtype=bool)
        measurementList = []
        hypothesisIndex = [0]
        # TODO:
        # http://stackoverflow.com/questions/15148496/python-passing-an-integer-by-reference
        for targetIndex in cluster:
            recActiveMeasurement(self.__targetList__[targetIndex],
                                 A1,
                                 measurementList,
                                 activeMeasurements,
                                 hypothesisIndex)
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
            if target.trackHypotheses is None:
                scoreArray.append(target.cumulativeNLLR)
            else:
                for hyp in target.trackHypotheses:
                    getTargetScore(hyp, scoreArray)

        scoreArray = []
        for targetIndex in cluster:
            getTargetScore(self.__targetList__[targetIndex], scoreArray)
        return scoreArray

    def _hypotheses2Nodes(self, selectedHypotheses, cluster):
        def recDFS(target, selectedHypothesis, nodeList, counter):
            if target.trackHypotheses is None:
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

    def _solveBLP_OR_TOOLS(self, A1, A2, f, nHyp):

        tic0 = time.time()
        nScores = len(f)
        (nMeas, nHyp) = A1.shape
        (nTargets, _) = A2.shape

        # Check matrix and vector dimension
        assert nScores == nHyp
        assert A1.shape[1] == A2.shape[1]

        # Initiate solver
        solver = pywraplp.Solver(
            'MHT-solver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Declare optimization variables
        tau = {i: solver.BoolVar("tau" + str(i)) for i in range(nHyp)}
        # tau = [solver.BoolVar("tau" + str(i)) for i in range(nHyp)]

        # Set objective
        solver.Minimize(solver.Sum([f[i] * tau[i] for i in range(nHyp)]))

        # <<< Problem child >>>
        tempMatrix = [[A1[row, col] * tau[col] for col in range(nHyp) if A1[row, col]]
                      for row in range(nMeas)]
        # <<< Problem child >>>
        toc0 = time.time() - tic0

        def setConstaints(solver, nMeas, nTargets, nHyp, tempMatrix, A2):
            # print("nMeas", nMeas)  # ~21
            # print("nTargets", nTargets)  # ~2
            # print("nHyp", nHyp)  # ~2788
            import numbers
            for row in range(nMeas):
                constraint = (solver.Sum(tempMatrix[row]) <= 1)
                solver.Add(constraint)

            for row in range(nTargets):
                solver.Add(solver.Sum([A2[row, col] * tau[col]
                                       for col in range(nHyp) if A2[row, col]]) == 1)

        tic1 = time.time()
        # Set constaints
        # import cProfile
        # cProfile.runctx("setConstaints(solver, nMeas, nTargets, nHyp, tempMatrix, A2)",
        #                 globals(), locals(), 'blpProfile.prof')
        setConstaints(solver, nMeas, nTargets, nHyp, tempMatrix, A2)
        toc1 = time.time() - tic1

        tic2 = time.time()
        # Solving optimization problem
        result_status = solver.Solve()
        assert result_status == pywraplp.Solver.OPTIMAL
        # print("Optim Time = ", solver.WallTime(), " milliseconds")
        toc2 = time.time() - tic2

        tic3 = time.time()
        selectedHypotheses = [i for i in range(nHyp)
                              if tau[i].solution_value() > 0.]
        assert len(selectedHypotheses) == nTargets
        toc3 = time.time() - tic3

        # print('_solveBLP_OR_TOOLS ({0:4.0f}|{1:4.0f}|{2:4.0f}|{3:4.0f}) ms = {4:4.0f}'.format(
        #     toc0 * 1000, toc1 * 1000, toc2 * 1000, toc3 * 1000, (toc0 + toc1 + toc2 + toc3) * 1000))
        # import pstats
        # p = pstats.Stats('blpProfile.prof')
        # p.strip_dirs().sort_stats('time').print_stats(20)
        # p.strip_dirs().sort_stats('cumulative').print_stats(20)
        return selectedHypotheses

    def _solveBLP_PULP(self, A1, A2, f, nHyp):

        tic0 = time.time()
        nScores = len(f)
        (nMeas, nHyp) = A1.shape
        (nTargets, _) = A2.shape

        # Check matrix and vector dimension
        assert nScores == nHyp
        assert A1.shape[1] == A2.shape[1]

        # Initialize solver
        prob = pulp.LpProblem("Association problem", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", range(nHyp), 0, 1, pulp.LpBinary)
        c = pulp.LpVariable.dicts("c", range(nHyp))
        for i in range(len(f)):
            c[i] = f[i]
        prob += pulp.lpSum(c[i] * x[i] for i in range(nHyp))
        toc0 = time.time() - tic0

        tic1 = time.time()
        for row in range(nMeas):
            prob += pulp.lpSum([A1[row, col] * x[col]
                                for col in range(nHyp) if A1[row, col]]) <= 1
        for row in range(nTargets):
            prob += pulp.lpSum([A2[row, col] * x[col]
                                for col in range(nHyp) if A2[row, col]]) == 1
        toc1 = time.time() - tic1

        tic2 = time.time()
        sol = prob.solve(self.solver)
        toc2 = time.time() - tic2

        tic3 = time.time()

        for threshold in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            selectedHypotheses = _getSelectedHyp2(prob, threshold)
            if len(selectedHypotheses) == nHyp:
                break
        toc3 = time.time() - tic3

        print('_solveBLP_PULP     ({0:4.0f}|{1:4.0f}|{2:4.0f}|{3:4.0f})ms = {4:4.0f}'.format(
            toc0 * 1000, toc1 * 1000, toc2 * 1000, toc3 * 1000, (toc0 + toc1 + toc2 + toc3) * 1000))
        return selectedHypotheses

    def _pruneTargetIndex(self, targetIndex, N):
        node = self.__trackNodes__[targetIndex]
        newRootNode = node.pruneDepth(N)
        if newRootNode != self.__targetList__[targetIndex]:
            self.__targetList__[targetIndex] = newRootNode
            self.__associatedMeasurements__[targetIndex] = self.__targetList__[
                targetIndex].getMeasurementSet()

    def _nScanPruning(self):
        for targetIndex, target in enumerate(self.__trackNodes__):
            self._pruneTargetIndex(targetIndex, self.__targetWindowSize__[targetIndex])

    def _pruneSimilarState(self, cluster, threshold):
        for targetIndex in cluster:
            leafParents = self.__targetList__[targetIndex].getLeafParents()
            for node in leafParents:
                node.pruneSimilarState(threshold)

    def _checkTrackerIntegrity(self):
        assert len(self.__trackNodes__) == len(self.__targetList__), \
            "There are not the same number trackNodes as targets"
        assert len(self.__targetList__) == len(set(self.__targetList__)), \
            "There are copies of targets in the target list"
        assert len(self.__trackNodes__) == len(set(self.__trackNodes__)), \
            "There are copies of track nodes in __trackNodes__"
        for target in self.__targetList__:
            target._checkScanNumberIntegrity()
            target._checkReferenceIntegrity()
        if len(self.__trackNodes__) > 0:
            assert len({node.scanNumber for node in self.__trackNodes__}) == 1, \
                "there are inconsistency in trackNodes scanNumber"
        scanNumber = len(self.__scanHistory__)
        for targetIndex, target in enumerate(self.__targetList__):
            leafNodes = target.getLeafNodes()
            for leafNode in leafNodes:
                assert leafNode.scanNumber == scanNumber, "Target " + str(targetIndex + 1)

    def plotValidationRegionFromRoot(self, stepsBack=1):
        def recPlotValidationRegionFromTarget(target, eta2, stepsBack):
            if target.trackHypotheses is None:
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
            if target.trackHypotheses is None:
                plt.plot([p.x() for p in newTrack], [p.y()
                                                     for p in newTrack], "--", **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    recPlotHypothesesTrack(hyp, newTrack, **kwargs)

        colors = kwargs.get("colors", self._getColorCycle())
        for target in self.__targetList__:
            recPlotHypothesesTrack(target, c=next(colors))

    def plotActiveTracks(self, **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for track in self.__trackNodes__:
            track.plotTrack(c=next(colors), **kwargs)

    def plotTerminatedTracks(self, **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for track in self.__terminatedTargets__:
            track.plotTrack(c=next(colors), markInitial=True, markEnd=True, **kwargs)

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
            if target.trackHypotheses is not None:
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
        fig = plt.gcf()
        size = fig.get_size_inches() * fig.dpi
        for i, initialTarget in enumerate(initialTargets):
            index = kwargs.get("index", list(range(len(initialTargets))))
            offset = 0.05 * size
            if len(index) != len(initialTargets):
                raise ValueError(
                    "plotInitialTargets: Need equal number of targets and indices")
            initialTarget.plotInitial(index=index[i], offset=offset)

    def _getColorCycle(self):
        return itertools.cycle(self.colors)

    def printTargetList(self, **kwargs):
        np.set_printoptions(precision=2, suppress=True)
        print("TargetList:")
        for targetIndex, target in enumerate(self.__targetList__):
            if kwargs.get("backtrack", False):
                print(target.stepBack().__str__(targetIndex=targetIndex))
            else:
                print(target.__str__(targetIndex=targetIndex))
        print()

    def printTimeLog(self):
        from termcolor import cprint
        self.toc *= 1000
        totalTime = self.toc[0]
        tooLong = totalTime > (self.period * 1000)
        nNodes = sum([target.getNumOfNodes() for target in self.__targetList__])
        nMeasurements = len(self.__scanHistory__[-1].measurements)
        scanNumber = len(self.__scanHistory__)
        nTargets = len(self.__targetList__)
        nClusters = len(self.__clusterList__)
        cprint(
            ('{:3.0f} '.format(scanNumber) +
             'nTrack {:2.0f}'.format(nTargets) + "  " +
             'Total {0:6.0f}'.format(totalTime) + "  " +
             'Process({0:3.0f}/{1:6.0f}) {2:6.1f}'.format(
                 nMeasurements, nNodes, self.toc[1]) + "  " +
             'Cluster({0:2.0f}) {1:5.1f}'.format(nClusters, self.toc[2]) + "  " +
             'Optim({0:g}) {1:6.1f}'.format(self.nOptimSolved, self.toc[3]) + "  " +
             # 'ILP-Prune {:5.0f}'.format(self.toc[4]) + "  " +
             'DynN {:4.1f}'.format(self.toc[5]) + " " +
             'N-Prune {:5.1f}'.format(self.toc[6]) + " " +
             'Kill {:3.1f}'.format(self.toc[7]) + " " +
             'Init {:5.1f}'.format(self.toc[8])
             ),
            'red' if tooLong else None,
            attrs=(['bold'] if tooLong else [])
        )


if __name__ == '__main__':
    pass
