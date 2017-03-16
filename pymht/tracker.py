"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Spring 2017
========================================================================================
"""
import pymht.utils.helpFunctions as hpf
import pymht.utils.kalman as kalman
import pymht.initiators.m_of_n as m_of_n
import pymht.models.pv as model
import time
import signal
import os
import logging
import datetime
import copy
import itertools
import functools
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from scipy.sparse.csgraph import connected_components
from ortools.linear_solver import pywraplp

# ----------------------------------------------------------------------------
# Instantiate logging object
# ----------------------------------------------------------------------------
cwd = os.getcwd()
logDir = os.path.join(cwd, 'logs')
if not os.path.exists(logDir):
    os.makedirs(logDir)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-25s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(logDir, 'myapp.log'),
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
log = logging.getLogger(__name__)
log.addHandler(console)


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
    def __init__(self, Phi, C, Gamma, P_d, P_0, R_RADAR, R_AIS, Q, lambda_phi,
                 lambda_nu, eta2, N, p0, radarRange, solverStr, **kwargs):

        log.debug('Running base main')

        self.logTime = kwargs.get("logTime", False)
        self.parallelize = kwargs.get("w", 1) > 1
        self.kwargs = kwargs

        if self.parallelize:
            self.nWorkers = max(kwargs.get("w") - 1, 1)
            self.workers = mp.Pool(self.nWorkers, initWorker)
        else:
            self.nWorkers = 0

        log.debug("Using " +
                  str(self.nWorkers + 1) +
                  " process(es) with " +
                  str(os.cpu_count()) +
                  " cores")

        # State space model
        self.A = Phi
        self.C = C
        self.Gamma = Gamma
        self.P_0 = P_0
        self.R_RADAR = R_RADAR
        self.R_AIS = R_AIS
        self.Q = Q

        # Target initiator
        self.maxSpeed = kwargs.get('maxSpeed', 20)
        self.M_required = kwargs.get('M_required', 2)
        self.N_checks = kwargs.get('N_checks', 3)
        self.mergeThreshold = 2*(model.sigmaR_RADAR_tracker**2)
        self.initiator = m_of_n.Initiator(self.M_required,
                                          self.N_checks,
                                          self.maxSpeed,
                                          self.C,
                                          self.R_RADAR,
                                          self.mergeThreshold,
                                          logLevel='DEBUG')

        # Tracker storage
        self.__targetList__ = []
        self.__targetWindowSize__ = []
        self.__scanHistory__ = []
        self.__associatedMeasurements__ = []
        self.__targetProcessList__ = []
        self.__trackNodes__ = np.empty(0, dtype=np.dtype(object))
        self.__terminatedTargets__ = []
        self.__clusterList__ = []
        self.__aisHistory__ = []

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
                           'ILP-Prune': np.array([0.0, 0]),
                           'DynN': np.array([0.0, 0]),
                           'N-Prune': np.array([0.0, 0]),
                           'Terminate': np.array([0.0, 0]),
                           'Init': np.array([0.0, 0]),
                           }
        if self.logTime:
            self.tic = {}
            self.toc = {}
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
        self.targetSizeLimit = 3000

        if ((kwargs.get("realTime") is not None) and
                (kwargs.get("realTime") is True)):
            _setHightPriority()

        # Misc
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        log.debug("Initiation done\n" + "#"*100 + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exeType, exeValue, traceback):
        if self.parallelize:
            self.workers.terminate()
            self.workers.join()

    def initiateTarget(self, newTarget):
        if newTarget.haveNoNeightbours(self.__targetList__, self.mergeThreshold):
            target = copy.copy(newTarget)
            target.scanNumber = len(self.__scanHistory__)
            target.P_d = self.default_P_d
            # assert target.measurementNumber is not None
            assert target.measurement is not None
            self.__targetList__.append(target)
            self.__associatedMeasurements__.append(set())
            self.__trackNodes__ = np.append(self.__trackNodes__, target)
            self.__targetWindowSize__.append(self.N)
        else:
            log.debug("Discarded an initial target: " + str(newTarget))

    def addMeasurementList(self, scanList, aisList=None, **kwargs):
        log.debug("addMeasurementList starting " + str(len(self.__scanHistory__) + 1))

        # Adding new data to history
        self.__scanHistory__.append(scanList)
        self.__aisHistory__.append(aisList)

        # Verifying time stamps
        scanTime = scanList.time
        log.debug('Radar time \t' + datetime.datetime.fromtimestamp(scanTime).strftime("%H:%M:%S.%f"))

        if aisList is not None:
            aisTime = aisList.time
            assert aisTime == scanTime
            log.debug('AIS time \t' + datetime.datetime.fromtimestamp(aisTime).strftime("%H:%M:%S.%f"))

        # 0 --Iterative procedure for tracking --
        self.tic['Total'] = time.time()

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 1 --Grow each track tree--
        if self.logTime:
            self.tic['Process'] = time.time()
        nMeas = len(scanList.measurements)
        measDim = self.C.shape[0]
        scanNumber = len(self.__scanHistory__)

        nTargets = len(self.__targetList__)
        timeSinceLastScan = scanTime - self.__scanHistory__[-1].time
        if not self.fixedPeriod:
            self.period = timeSinceLastScan

        unused_measurement_indices = np.ones(nMeas, dtype=np.bool)

        if self.logTime:
            self.leafNodeTimeList = []

        targetProcessTimes = np.zeros(nTargets)
        nTargetNodes = np.zeros(nTargets)
        for targetIndex, target in enumerate(self.__targetList__):
            if self.parallelize:
                # try openMP
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
                            scanList,
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
                        scanList,
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
                    nTargetNodes[targetIndex] = nNodes
                    dummyNodesData, radarNodesData, fusedNodesData = self._processLeafNodes(targetNodes,
                                                                                            scanList,
                                                                                            aisList)
                    x_bar_list, P_bar_list = dummyNodesData
                    gated_x_hat_list, P_hat_list, gatedIndicesList, nllrList = radarNodesData
                    (fused_x_hat_list,
                     fused_P_hat_list,
                     fused_radar_indices_list,
                     fused_nllr_list,
                     fused_mmsi_list) = fusedNodesData
                    gatedMeasurementsList = [np.array(scanList.measurements[gatedIndices])
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
                                           scanList.measurements,
                                           gated_x_hat_list[i],
                                           P_hat_list[i],
                                           nllrList[i],
                                           (fused_x_hat_list[i],
                                            fused_P_hat_list[i],
                                            fused_radar_indices_list[i],
                                            fused_nllr_list[i],
                                            fused_mmsi_list[i]))

                    for i in range(nNodes):
                        for measurementIndex in gatedIndicesList[i]:
                            self.__associatedMeasurements__[targetIndex].update(
                                {(scanNumber, measurementIndex + 1)})
                    targetProcessTimes[targetIndex] = time.time() - tic

        if self.logTime:
            self.toc['Process'] = time.time() - self.tic['Process']
        if self.parallelize:
            log.debug(str([round(e) for e in self.leafNodeTimeList]))
        if kwargs.get("printAssociation", False):
            print(*self.__associatedMeasurements__, sep="\n", end="\n\n")

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 2 --Cluster targets --
        if self.logTime:
            self.tic['Cluster'] = time.time()
        self.__clusterList__ = self._findClustersFromSets()
        if self.logTime:
            self.toc['Cluster'] = time.time() - self.tic['Cluster']
        if kwargs.get("printCluster", False):
            hpf.printClusterList(self.__clusterList__)

        # 3 --Maximize global (cluster vise) likelihood--
        if self.logTime:
            self.tic['Optim'] = time.time()
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
            self.toc['Optim'] = time.time() - self.tic['Optim']

        # 4 -- ILP Pruning
        if self.logTime:
            self.tic['ILP-Prune'] = time.time()

        if self.logTime:
            self.toc['ILP-Prune'] = time.time() - self.tic['ILP-Prune']

        # 5 -- Dynamic window size
        if self.logTime:
            self.tic['DynN'] = time.time()

        totalGrowTime = sum(targetProcessTimes)
        growTimeLimit = self.period * 0.5
        tooSlowTotal = totalGrowTime > growTimeLimit
        targetProcessTimeLimit = growTimeLimit / len(self.__targetList__) if tooSlowTotal else 200e-3
        for targetIndex, target in enumerate(self.__targetList__):
            targetProcessTime = targetProcessTimes[targetIndex]
            targetSize = target.getNumOfNodes()
            tooSlow = targetProcessTime > targetProcessTimeLimit
            tooLarge = targetSize > self.targetSizeLimit
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
                log.debug(infoString)

        tempTotalTime = time.time() - self.tic['Total']
        if tempTotalTime > (self.period * 0.8):
            self.N = max(1, self.N - 1)
            log.debug(
                'Iteration took to long time ({0:.1f}ms), reducing window size roof from {1:} to  {2:}'.format(
                    tempTotalTime * 1000, self.N + 1, self.N))
            self.__targetWindowSize__ = [min(e, self.N) for e in self.__targetWindowSize__]

        if self.logTime:
            self.toc['DynN'] = time.time() - self.tic['DynN']

        # 5 --Prune sliding window --
        if self.logTime:
            self.tic['N-Prune'] = time.time()
        self._nScanPruning()
        if self.logTime:
            self.toc['N-Prune'] = time.time() - self.tic['N-Prune']

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 6 -- Pick out dead tracks (terminate)
        if self.logTime:
            self.tic['Terminate'] = time.time()
        deadTracks = []
        for trackIndex, trackNode in enumerate(self.__trackNodes__):
            # Check outside range
            if trackNode.isOutsideRange(self.position.array, self.range):
                deadTracks.append(trackIndex)
                log.debug("Terminating track {:} since it is out of range".format(trackIndex))

            # Check if track is to insecure
            elif trackNode.cumulativeNLLR > self.NLLR_UPPER_LIMIT:
                deadTracks.append(trackIndex)
                log.debug("Terminating track {:} since its cost is above the threshold".format(trackIndex))

        self._terminateTracks(deadTracks)
        if self.logTime:
            self.toc['Terminate'] = time.time() - self.tic['Terminate']

        # 7 -- Initiate new tracks
        if self.logTime:
            self.tic['Init'] = time.time()
        unused_measurements = scanList.filterUnused(unused_measurement_indices)
        # import cProfile, pstats
        # new_initial_targets = []
        # cProfile.runctx("new_initial_targets = self.initiator.processMeasurements(unused_measurements)",
        #                 globals(), locals(), 'profile/initiatorProfile.prof')
        # p = pstats.Stats('profile/initiatorProfile.prof')
        # p.strip_dirs().sort_stats('time').print_stats(20)
        # p.strip_dirs().sort_stats('cumulative').print_stats(20)
        new_initial_targets = self.initiator.processMeasurements(unused_measurements)

        for initial_target in new_initial_targets:
            log.debug("\tNew target({}): ".format(len(self.__targetList__) + 1) + str(initial_target))
            self.initiateTarget(initial_target)
        if self.logTime:
            self.toc['Init'] = time.time() - self.tic['Init']

        self.toc['Total'] = time.time() - self.tic['Total']

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        if self.logTime:
            for k, v in self.runtimeLog.items():
                if k in self.toc:
                    v += np.array([self.toc[k], 1])

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

        if nTargetNodes.size > 0:
            avgTimePerNode = self.toc['Process']*1e6 / np.sum(nTargetNodes)
            log.debug("Process time per (old) leaf node = {:.0f}us".format(avgTimePerNode))
        log.debug("addMeasurement completed \n" + self.getTimeLogString() + "\n")

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

    def _processLeafNodes(self, targetNodes, scanList, aisList):
        dummyNodesData = self.__predictDummyMeasurements(targetNodes)

        gatedRadarData = self.__processMeasurements(targetNodes,
                                                    scanList,
                                                    dummyNodesData,
                                                    model.C_RADAR,
                                                    self.R_RADAR)

        newNodesData = self.__createPureRadarNodes(gatedRadarData)

        gatedAisData = self.__processMeasurements(targetNodes,
                                                  aisList,
                                                  dummyNodesData,
                                                  model.C_AIS,
                                                  self.R_AIS)

        fusedNodesData = self.__fuseRadarAndAis(targetNodes,
                                                gatedRadarData,
                                                gatedAisData)

        return dummyNodesData, newNodesData, fusedNodesData

    def __createPureRadarNodes(self, gatedRadarData):
        (gated_radar_indices_list,
         _,
         gated_x_radar_hat_list,
         P_radar_hat_list,
         _,
         _,
         radar_nllr_list) = gatedRadarData

        # (gated_x_hat_list, P_hat_list, gatedIndicesList, nllrList)
        newNodesData = (gated_x_radar_hat_list,
                        P_radar_hat_list,
                        gated_radar_indices_list,
                        radar_nllr_list)
        assert all(d is not None for d in newNodesData)
        return newNodesData

    def __fuseRadarAndAis(self, targetNodes, gatedRadarData, gatedAisData):
        nNodes = len(targetNodes)
        if gatedAisData is None:
            return ([np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)])

        (gated_radar_indices_list,
         gated_z_radar_tilde_list,
         gated_x_radar_hat_list,
         P_radar_hat_list,
         S_radar_list,
         radar_nis,
         radar_nllr_list) = gatedRadarData

        (gated_ais_indices_list,
         gated_z_ais_tilde_list,
         gated_x_ais_hat_list,
         P_ais_hat_list,
         S_ais_list,
         ais_nis,
         ais_nllr_list) = gatedAisData

        # print("Processing nodes:", *targetNodes, sep="\n")
        # print("gated_radar_indices_list", gated_radar_indices_list)
        # print("Fusing AIS and radar")
        # print("gated_ais_indices_list", gated_ais_indices_list)
        # print("gated_z_ais_tilde_list", gated_z_ais_tilde_list)
        # print("gated_x_ais_hat_list", gated_x_ais_hat_list)
        # print("P_ais_hat_list\n", P_ais_hat_list)
        # print("S_ais_list\n", S_ais_list)
        # print("ais_nis", ais_nis)
        # print("ais_nllr_list", ais_nllr_list)

        fused_x_hat_list = []
        fused_P_hat_list = []
        fused_radar_indices_list = []
        fused_nllr_list = []
        fused_mmsi_list = []

        for i in range(nNodes):
            x_hat_list = []
            radar_indices_list = []
            nllr_list = []
            mmsi_list = []
            fusedCovariance = P_ais_hat_list[i]
            for j, radarMeasurementIndex in enumerate(gated_radar_indices_list[i]):
                # assert gated_x_ais_hat_list[i].shape[0] == gated_radar_indices_list[i].shape[0], \
                #     str(gated_x_ais_hat_list[i].shape) + str(gated_radar_indices_list[i].shape)
                for k, aisMeasurementIndex in enumerate(gated_ais_indices_list[i]):
                    # print("i", i)
                    # print("j", j)
                    # print("k", k)
                    # print("radarMeasurementIndex", radarMeasurementIndex)
                    # print("aisMeasurementIndex", aisMeasurementIndex)
                    fusedState = gated_x_ais_hat_list[i][k]
                    # print("fusedState", fusedState)
                    fusedNLLR = ais_nllr_list[i][k] + radar_nllr_list[i][k]
                    mmsi = self.__aisHistory__[-1].measurements[aisMeasurementIndex].mmsi
                    x_hat_list.append(fusedState)
                    radar_indices_list.append(radarMeasurementIndex)
                    nllr_list.append(fusedNLLR)
                    mmsi_list.append(mmsi)
            fused_x_hat_list.append(np.array(x_hat_list, ndmin=2))
            fused_P_hat_list.append(fusedCovariance)
            fused_radar_indices_list.append(np.array(radar_indices_list))
            fused_nllr_list.append(np.array(nllr_list))
            fused_mmsi_list.append(np.array(mmsi_list))

        assert len(fused_x_hat_list) == nNodes
        assert len(fused_P_hat_list) == nNodes
        assert len(fused_radar_indices_list) == nNodes
        assert len(fused_nllr_list) == nNodes
        assert len(fused_mmsi_list) == nNodes
        # print("nNodes", nNodes)
        for i in range(nNodes):
            # print("Hei")
            # print(type(fused_x_hat_list[i]))
            assert fused_x_hat_list[i].ndim == 2, str(fused_x_hat_list[i].ndim)
            nFusedNodes, nStates = fused_x_hat_list[i].shape
            # print(fused_x_hat_list[i])
            # print("nFusedNodes, nStates", nFusedNodes, nStates)
            if nStates == 0: continue
            assert fused_P_hat_list[i].shape == (nStates, nStates), str(fused_P_hat_list[i].shape)

        fusedNodesData = (fused_x_hat_list,
                          fused_P_hat_list,
                          fused_radar_indices_list,
                          fused_nllr_list,
                          fused_mmsi_list)

        return fusedNodesData

    def __processMeasurements(self, targetNodes, measurementList, dummyNodesData, C, R):
        if measurementList is None: return None
        nNodes = len(targetNodes)
        nMeas = len(measurementList.measurements)
        meas_dim = C.shape[0]
        x_bar_list, P_bar_list = dummyNodesData

        nodesPredictionData = self.__predictPrecalcBulk(targetNodes, C, R, dummyNodesData)

        (z_hat_list,
         S_list,
         S_inv_list,
         K_list,
         P_hat_list) = nodesPredictionData

        z_list = measurementList.getMeasurements()
        assert z_list.shape[1] == meas_dim

        z_tilde_list = kalman.z_tilde(z_list, z_hat_list, nNodes, meas_dim)
        assert z_tilde_list.shape == (nNodes, nMeas, meas_dim)

        nis = kalman.normalizedInnovationSquared(z_tilde_list, S_inv_list)
        assert nis.shape == (nNodes, nMeas,)

        gated_filter = nis <= self.eta2
        assert gated_filter.shape == (nNodes, nMeas)

        gated_indices_list = [np.nonzero(gated_filter[row])[0]
                              for row in range(nNodes)]
        assert len(gated_indices_list) == nNodes

        gated_z_tilde_list = [z_tilde_list[i, gated_indices_list[i]]
                              for i in range(nNodes)]
        assert len(gated_z_tilde_list) == nNodes
        assert all([z_tilde.shape[1] == meas_dim for z_tilde in gated_z_tilde_list])

        gated_x_hat_list = [kalman.numpyFilter(
            x_bar_list[i], K_list[i], gated_z_tilde_list[i])
                            for i in range(nNodes)]
        assert len(gated_x_hat_list) == nNodes

        nllr_list = [kalman.nllr(self.lambda_ex,
                                 targetNodes[i].P_d,
                                 S_list[i],
                                 nis[i, gated_filter[i]])
                     for i in range(nNodes)]
        assert len(nllr_list) == nNodes

        return (gated_indices_list,
                gated_z_tilde_list,
                gated_x_hat_list,
                P_hat_list,
                S_list,
                nis,
                nllr_list)

    def __predictDummyMeasurements(self, targetNodes):
        nNodes = len(targetNodes)
        radarMeasDim, nStates = model.C_RADAR.shape
        x_0_list = np.array([target.x_0 for target in targetNodes],
                            ndmin=2)
        P_0_list = np.array([target.P_0 for target in targetNodes],
                            ndmin=3)
        assert x_0_list.shape == (nNodes, nStates)
        assert P_0_list.shape == (nNodes, nStates, nStates)

        x_bar_list, P_bar_list = kalman.predict(self.A, self.Q, self.Gamma, x_0_list, P_0_list)
        return x_bar_list, P_bar_list

    def __predictPrecalcBulk(self, targetNodes, C, R, dummyNodesData):
        nNodes = len(targetNodes)
        measDim, nStates = C.shape
        x_bar_list, P_bar_list = dummyNodesData

        z_hat_list, S_list, S_inv_list, K_list, P_hat_list = kalman.precalc(
            self.A, C, self.Q, R, self.Gamma, x_bar_list, P_bar_list)

        assert S_list.shape == (nNodes, measDim, measDim)
        assert S_inv_list.shape == (nNodes, measDim, measDim)
        assert K_list.shape == (nNodes, nStates, measDim)
        assert P_hat_list.shape == P_bar_list.shape
        assert z_hat_list.shape == (nNodes, measDim)

        return z_hat_list, S_list, S_inv_list, K_list, P_hat_list

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
        # log.debug("matricesTime\t" + str(round(t1,3)))

        selectedHypotheses = self._solveBLP_OR_TOOLS(A1, A2, C, len(cluster))
        # selectedHypotheses = self._solveBLP_PULP(A1, A2, C_RADAR, len(cluster))
        # assert selectedHypotheses0 == selectedHypotheses
        selectedNodes = self._hypotheses2Nodes(selectedHypotheses, cluster)
        selectedNodesArray = np.array(selectedNodes)
        # log.debug("Solving optimal association in cluster with targets" +
        #                str(cluster) + ",   \t" +
        #                str(sum(nHypInClusterArray)) + " hypotheses and " +
        #                str(nRealMeasurementsInCluster) + " real measurements.")
        # log.debug("nHypothesesInCluster {:}".format(sum(nHypInClusterArray)))
        # log.debug("nRealMeasurementsInCluster {:}".format(nRealMeasurementsInCluster))
        # log.debug("nTargetsInCluster {:}".format(len(cluster)))
        # log.debug("nHypInClusterArray {:}".format(HypInClusterArray))
        # log.debug("c =" + str(c))
        # log.debug("A1 \n" + str(A1))
        # log.debug("size(A1) " + str(A1.shape) + "\t=>\t" + str(nRealMeasurementsInCluster * sum(nHypInClusterArray)))
        # log.debug("A2 \n" + str(A2))
        # log.debug("measurementList" + str(measurementList))
        # log.debug("selectedHypotheses" + str(selectedHypotheses))
        # log.debug("selectedNodes" +  str(*selectedNodes))
        # log.debug("selectedNodesArray" + str(*selectedNodesArray))

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

        A1 = np.zeros((nRow, nCol), dtype=bool)
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
            for row in range(nMeas):
                constraint = (solver.Sum(tempMatrix[row]) <= 1)
                solver.Add(constraint)

            for row in range(nTargets):
                solver.Add(solver.Sum([A2[row, col] * tau[col]
                                       for col in range(nHyp) if A2[row, col]]) == 1)

        tic1 = time.time()
        setConstaints(solver, nMeas, nTargets, nHyp, tempMatrix, A2)
        toc1 = time.time() - tic1

        tic2 = time.time()
        # Solving optimization problem
        result_status = solver.Solve()
        assert result_status == pywraplp.Solver.OPTIMAL
        log.debug("Optim Time = " + str(solver.WallTime()) + " milliseconds")
        toc2 = time.time() - tic2

        tic3 = time.time()
        selectedHypotheses = [i for i in range(nHyp)
                              if tau[i].solution_value() > 0.]
        assert len(selectedHypotheses) == nTargets
        toc3 = time.time() - tic3

        log.debug('_solveBLP_OR_TOOLS ({0:4.0f}|{1:4.0f}|{2:4.0f}|{3:4.0f}) ms = {4:4.0f}'.format(
            toc0 * 1000, toc1 * 1000, toc2 * 1000, toc3 * 1000, (toc0 + toc1 + toc2 + toc3) * 1000))
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

        log.debug('_solveBLP_PULP     ({0:4.0f}|{1:4.0f}|{2:4.0f}|{3:4.0f})ms = {4:4.0f}'.format(
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

    def getSmoothTracks(self):
        return [track.getSmoothTrack() for track in self.__trackNodes__]

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
                plt.plot([p.x() for p in newTrack],
                         [p.y() for p in newTrack],
                         "--",
                         **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    recPlotHypothesesTrack(hyp, newTrack, **kwargs)

        colors = kwargs.get("colors", self._getColorCycle())
        for target in self.__targetList__:
            recPlotHypothesesTrack(target, c=next(colors))
        if kwargs.get('markStates', False):
            self.plotStatesFromRoot(dummy=True, real=True, includeHistory=False)
            # tracker.plotValidationRegionFromRoot() # TODO: Does not work

    def plotActiveTracks(self, **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for i, track in enumerate(self.__trackNodes__):
            track.plotTrack(root=self.__targetList__[i], c=next(colors), period = self.period, **kwargs)
        if kwargs.get('markStates', True):
            defaults = {'labels': False, 'dummy': True, 'real': True, 'ais':True}
            self.plotStatesFromTracks(**{**defaults, **kwargs})

    def plotTerminatedTracks(self, **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for track in self.__terminatedTargets__:
            track.plotTrack(c=next(colors), markInitial=True, markEnd=True, **kwargs)

    def plotMeasurementsFromTracks(self, stepsBack=float('inf'), **kwargs):
        for node in self.__trackNodes__:
            node.plotMeasurement(stepsBack, **kwargs)

    def plotStatesFromTracks(self, stepsBack=float('inf'), **kwargs):
        for node in self.__trackNodes__:
            node.plotStates(stepsBack, **kwargs)

    def plotMeasurementsFromRoot(self, **kwargs):
        if not (("real" in kwargs) or ("dummy" in kwargs)):
            return
        plottedMeasurements = set()
        for target in self.__targetList__:
            if kwargs.get("includeHistory", False):
                target.getInitial().recPlotMeasurements(plottedMeasurements, **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    hyp.recPlotMeasurements(plottedMeasurements, **kwargs)

    def plotStatesFromRoot(self, **kwargs):
        if not (("real" in kwargs) or ("dummy" in kwargs)):
            return
        for target in self.__targetList__:
            if kwargs.get("includeHistory", False):
                target.getInitial().recPlotStates(**kwargs)
            else:
                for hyp in target.trackHypotheses:
                    hyp.recPlotStates(**kwargs)

    def plotScanIndex(self, index, **kwargs):
        self.__scanHistory__[index].plot(**kwargs)

    def plotLastScan(self, **kwargs):
        self.__scanHistory__[-1].plot(**kwargs)

    def plotLastAisUpdate(self, **kwargs):
        if self.__aisHistory__[-1] is not None:
            self.__aisHistory__[-1].plot(**kwargs)

    def plotAllScans(self, **kwargs):
        for scan in self.__scanHistory__:
            scan.plot(**kwargs)

    def plotAllAisUpdates(self, **kwargs):
        for update in self.__aisHistory__:
            if update is not None:
                update.plot(**kwargs)

    def plotVelocityArrowForTrack(self, stepsBack=1):
        for track in self.__trackNodes__:
            track.plotVelocityArrow(stepsBack)

    def plotInitialTargets(self, **kwargs):
        initialTargets = [target.getInitial() for target in self.__targetList__]
        fig = plt.gcf()
        size = fig.get_size_inches() * fig.dpi
        for i, initialTarget in enumerate(initialTargets):
            index = kwargs.get("index", list(range(len(initialTargets))))
            offset = 0.05 * size
            if len(index) != len(initialTargets):
                raise ValueError(
                    "plotInitialTargets: Need equal number of targets and indices")
            initialTarget.markInitial(index=index[i], offset=offset)

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

    def getTimeLogString(self):
        tocMS = {k: v * 1000 for k, v in self.toc.items()}
        totalTime = tocMS['Total']
        nNodes = sum([target.getNumOfNodes() for target in self.__targetList__])
        nMeasurements = len(self.__scanHistory__[-1].measurements)
        nAisUpdates = len(self.__aisHistory__[-1].measurements) if self.__aisHistory__[-1] is not None else 0
        scanNumber = len(self.__scanHistory__)
        nTargets = len(self.__targetList__)
        nClusters = len(self.__clusterList__)
        timeLogString = ('{:<3.0f} '.format(scanNumber) +
                         'nTrack {:2.0f}'.format(nTargets) + "  " +
                         'Total {0:6.0f}'.format(totalTime) + "  " +
                         'Process({0:3.0f}+{1:<3.0f}/{2:6.0f}) {3:6.1f}'.format(
                             nMeasurements, nAisUpdates, nNodes, tocMS['Process']) + "  " +
                         'Cluster({0:2.0f}) {1:5.1f}'.format(nClusters, tocMS['Cluster']) + "  " +
                         'Optim({0:g}) {1:6.1f}'.format(self.nOptimSolved, tocMS['Optim']) + "  " +
                         # 'ILP-Prune {:5.0f}'.format(self.toc['ILP-Prune']) + "  " +
                         'DynN {:4.1f}'.format(tocMS['DynN']) + " " +
                         'N-Prune {:5.1f}'.format(tocMS['N-Prune']) + " " +
                         'Kill {:3.1f}'.format(tocMS['Terminate']) + " " +
                         'Init {:5.1f}'.format(tocMS['Init']))
        return timeLogString

    def printTimeLog(self):
        from termcolor import cprint
        tooLong = self.toc['Total'] > self.period
        cprint(self.getTimeLogString(),
               'red' if tooLong else None,
               attrs=(['bold'] if tooLong else [])
               )


if __name__ == '__main__':
    pass
