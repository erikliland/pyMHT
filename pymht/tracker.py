"""
========================================================================================
TRACK-ORIENTED-(MULTI-TARGET)-MULTI-HYPOTHESIS-TRACKER (with Kalman Filter and PV-model)
by Erik Liland, Norwegian University of Science and Technology
Trondheim, Norway
Spring 2017
========================================================================================
"""
import matplotlib
matplotlib.use('Agg')
from pymht.utils.xmlDefinitions import *
from pymht.pyTarget import Target
import pymht.utils.kalman as kalman
import pymht.initiators.m_of_n as m_of_n
import pymht.models.pv as pv
import pymht.models.ais as ais_model
import time
import copy
import logging
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from ortools.linear_solver import pywraplp
from termcolor import cprint
import xml.etree.ElementTree as ET
from .utils.classDefinitions import AisMessageList
import os
npVersionTuple = np.__version__.split('.')
assert (int(npVersionTuple[0]) >= 1 and int(npVersionTuple[1]) >= 12), str(np.__version__)

# ----------------------------------------------------------------------------
# Instantiate logging object
# -------------------------------------------------------1---------------------

log = logging.getLogger(__name__)


class Tracker():

    def __init__(self, model, radarPeriod, lambda_phi, lambda_nu, **kwargs):

        # Radar parameters
        self.position = kwargs.get('position', np.array([0., 0.]))
        self.radarRange = kwargs.get('radarRange', float('inf'))
        self.radarPeriod = radarPeriod
        self.totalGrowTimeLimit = self.radarPeriod * 0.5
        self.nodeGrowTimeLimit = 200e-3
        self.fixedPeriod = True
        self.default_P_d = kwargs.get('P_d', 0.8)
        assert self.default_P_d < 1 and self.default_P_d > 0, "Invalid P_d"

        # State space pv
        self.A = model.Phi(radarPeriod)
        self.C = model.C_RADAR
        self.P_0 = model.P0
        self.R_RADAR = model.R_RADAR()
        # self.R_AIS = model.R_AIS()
        self.Q = model.Q(radarPeriod)

        # Target initiator
        self.maxSpeedMS = kwargs.get('maxSpeedMS', 20)
        self.M_required = kwargs.get('M_required', 2)
        self.N_checks = kwargs.get('N_checks', 3)
        self.mergeThreshold = 4 * (model.sigmaR_RADAR_tracker ** 2)
        self.initiator = m_of_n.Initiator(self.M_required,
                                          self.N_checks,
                                          self.maxSpeedMS,
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
        self.trackIdCounter = 0

        # Timing and logging
        self.runtimeLog = {'Total': [],
                           'Process': [],
                           'Cluster': [],
                           'Optim': [],
                           'ILP-Prune': [],
                           'DynN': [],
                           'N-Prune': [],
                           'Terminate': [],
                           'Init': [],
                           }
        self.tic = {}
        self.toc = {}
        self.nOptimSolved = 0
        self.leafNodeTimeList = []
        self.createComputationTime = None

        # Tracker parameters
        self.pruneSimilar = kwargs.get('pruneSimilar', False)
        self.lambda_phi = lambda_phi
        self.lambda_nu = lambda_nu
        self.lambda_ex = lambda_phi + lambda_nu
        self.P_r = 0.95
        self.P_ais = 0.5
        self.eta2 = kwargs.get('eta2', 5.99)
        self.eta2_ais = kwargs.get('eta2_ais', 9.45)
        N = kwargs.get('N', 5)
        self.N_max = copy.copy(N)
        self.N = copy.copy(N)
        self.scoreUpperLimit = -np.log(1 - self.default_P_d) * 0.8
        self.clnnrUpperLimit = 3.0
        self.pruneThreshold = kwargs.get("pruneThreshold", 4)
        self.targetSizeLimit = 3000

        if ((kwargs.get("realTime") is not None) and
                (kwargs.get("realTime") is True)):
            self.setHighPriority()

        # Misc
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        log.info("Initiation done\n" + "#" * 100)

    def setHighPriority(self):
        import psutil
        import platform
        p = psutil.Process(os.getpid())
        OS = platform.system()
        if (OS == "Darwin") or (OS == "Linux"):
            p.nice(5)
        elif OS == "Windows":
            p.nice(psutil.HIGH_PRIORITY_CLASS)

    def preInitialize(self, simList):
        for initialTarget in simList[0]:
            self.initiateTarget(Target(initialTarget.time,
                                       None,
                                       initialTarget.cartesianState(),
                                       pv.P0,
                                       status=preinitializedTag))

    def initiateTarget(self, newTarget):
        if newTarget.haveNoNeightbours(self.__targetList__, self.mergeThreshold):
            target = copy.copy(newTarget)
            target.scanNumber = len(self.__scanHistory__)
            target.P_d = self.default_P_d
            target.ID = copy.copy(self.trackIdCounter)
            target.isRoot = True
            self.trackIdCounter += 1
            self.__targetList__.append(target)
            self.__associatedMeasurements__.append(set())
            self.__trackNodes__ = np.append(self.__trackNodes__, target)
            self.__targetWindowSize__.append(self.N)
        else:
            log.debug("Discarded an initial target: " + str(newTarget))

    def addMeasurementList(self, scanList, aisList=AisMessageList(), **kwargs):
        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()
        self.tic.clear()
        self.toc.clear()

        log.info("addMeasurementList starting " + str(len(self.__scanHistory__) + 1))

        # Adding new data to history
        self.__scanHistory__.append(scanList)
        self.__aisHistory__.append(aisList)

        # Verifying time stamps
        scanTime = scanList.time
        log.debug('Radar time \t' +
                  datetime.datetime.fromtimestamp(scanTime).strftime("%H:%M:%S.%f"))

        if aisList is not None:
            assert all([float(aisMessage.time) < scanTime for aisMessage in aisList])
            assert all([float(aisMessage.time) > scanTime - self.radarPeriod for aisMessage in aisList]),\
                str(scanTime) + str([m.time for m in aisList])
            mmsiList = [m.mmsi for m in aisList]
            mmsiSet = set(mmsiList)
            assert len(mmsiList) == len(mmsiSet), "Duplicate MMSI in aisList"
            log.debug(
                'AIS times \t' + ','.join([aisMeasurement.getTimeString() for aisMeasurement in aisList]))
            log.debug("AIS list:\n" + '\n'.join([str(m) for m in aisList]))
            nAisMeasurements = len(aisList)

        # 0 --Iterative procedure for tracking --
        self.tic['Total'] = time.time()

        # 1 --Grow each track tree--
        self.tic['Process'] = time.time()
        nRadarMeas = len(scanList.measurements)
        radarMeasDim = self.C.shape[0]
        scanNumber = len(self.__scanHistory__)
        nTargets = len(self.__targetList__)
        timeSinceLastScan = scanTime - self.__scanHistory__[-1].time
        if not self.fixedPeriod:
            self.radarPeriod = timeSinceLastScan
        unusedRadarMeasurementIndices = np.ones(nRadarMeas, dtype=np.bool)
        self.leafNodeTimeList = []
        targetProcessTimes = np.zeros(nTargets)
        nTargetNodes = np.zeros(nTargets)
        for targetIndex, _ in enumerate(self.__targetList__):
            self._growTarget(targetIndex, nTargetNodes, scanList, aisList, radarMeasDim,
                             unusedRadarMeasurementIndices, scanTime, scanNumber, targetProcessTimes)
        self.toc['Process'] = time.time() - self.tic['Process']

        if kwargs.get("printAssociation", False):
            print(*self.__associatedMeasurements__, sep="\n", end="\n\n")

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 2 --Cluster targets --
        self.tic['Cluster'] = time.time()
        self.__clusterList__ = self._findClustersFromSets()
        self.toc['Cluster'] = time.time() - self.tic['Cluster']
        if kwargs.get("printCluster", False):
            self.printClusterList(self.__clusterList__)

        # 3 --Maximize global (cluster vise) likelihood--
        self.tic['Optim'] = time.time()
        self.nOptimSolved = 0
        for cluster in self.__clusterList__:
            if len(cluster) == 1:
                if kwargs.get('pruneSimilar', False):
                    self._pruneSimilarState(cluster, self.pruneThreshold)
                self.__trackNodes__[cluster] = self.__targetList__[
                    cluster[0]]._selectBestHypothesis()
            else:
                self.__trackNodes__[cluster] = self._solveOptimumAssociation(cluster)
                self.nOptimSolved += 1
        self.toc['Optim'] = time.time() - self.tic['Optim']

        # 4 -- ILP Pruning
        self.tic['ILP-Prune'] = time.time()
        # Not implemented
        self.toc['ILP-Prune'] = time.time() - self.tic['ILP-Prune']

        # 5 -- Dynamic window size
        self.tic['DynN'] = time.time()
        if kwargs.get('dynamicWindow', False):
            self.__dynamicWindow(targetProcessTimes)
        self.toc['DynN'] = time.time() - self.tic['DynN']

        # 6 -- Pick out dead tracks (terminate)
        self.tic['Terminate'] = time.time()
        deadTracks = self.__analyzeTrackTermination()
        self._terminateTracks(deadTracks)
        self.toc['Terminate'] = time.time() - self.tic['Terminate']

        # 5 --Prune sliding window --
        self.tic['N-Prune'] = time.time()
        self._nScanPruning()
        self.toc['N-Prune'] = time.time() - self.tic['N-Prune']

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        # 7 -- Initiate new tracks
        self.tic['Init'] = time.time()
        unusedRadarMeasurements = scanList.filterUnused(unusedRadarMeasurementIndices)
        usedAisMmsi = [[a[1] for a in targetAssociations if a[0] == scanNumber and a[1]>=1e8]
                       for targetAssociations in self.__associatedMeasurements__]
        usedAisMmsi = [item for sublist in usedAisMmsi for item in sublist]
        unusedAisMeasurements = aisList.filterUnused(set(usedAisMmsi))
        if not kwargs.get('aisInitialization', True):
            unusedAisMeasurements = []
        new_initial_targets = self.initiator.processMeasurements(unusedRadarMeasurements, unusedAisMeasurements)
        for initial_target in new_initial_targets:
            log.info("\tNew target({}): ".format(
                len(self.__targetList__) + 1) + str(initial_target))
            self.initiateTarget(initial_target)
        self.toc['Init'] = time.time() - self.tic['Init']

        # Logging critical time constraints
        self.toc['Total'] = time.time() - self.tic['Total']
        if self.toc['Total'] > self.radarPeriod:
            log.critical("Did not pass real time demand! Used {0:.0f}ms of {1:.0f}ms".format(
                self.toc['Total'] * 1000, self.radarPeriod * 1000))
        elif self.toc['Total'] > self.radarPeriod * 0.6:
            log.warning("Did almost not pass real time demand! Used {0:.0f}ms of {1:.0f}ms".format(
                self.toc['Total'] * 1000, self.radarPeriod * 1000))

        if kwargs.get("checkIntegrity", False):
            self._checkTrackerIntegrity()

        for k, v in self.runtimeLog.items():
            if k in self.toc:
                v.append(self.toc[k])

        if kwargs.get("printInfo", False):
            print("Added scan number:", len(self.__scanHistory__),
                  " \tnRadarMeas ", nRadarMeas, sep="")

        if kwargs.get("printTime", False):
            self.printTimeLog(**kwargs)

        if nTargetNodes.size > 0:
            avgTimePerNode = self.toc['Process'] * 1e6 / np.sum(nTargetNodes)
            log.debug(
                "Process time per (old) leaf node = {:.0f}us".format(avgTimePerNode))
        log.info("addMeasurement completed \n" + self.getTimeLogString() + "\n")

    def _growTarget(self, targetIndex, nTargetNodes, scanList, aisList, measDim, unused_measurement_indices,
                    scanTime, scanNumber, targetProcessTimes):
        tic = time.time()
        target = self.__targetList__[targetIndex]
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
            node.spawnNewNodes(self.__associatedMeasurements__[targetIndex],
                               scanTime,
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

        targetProcessTimes[targetIndex] = time.time() - tic

    def _terminateTracks(self, deadTracks):
        deadTracks.sort(reverse=True)
        for trackIndex in deadTracks:
            nTargetPre = len(self.__targetList__)
            nTracksPre = self.__trackNodes__.shape[0]
            nAssociationsPre = len(self.__associatedMeasurements__)
            targetListTypePre = type(self.__targetList__)
            trackListTypePre = type(self.__trackNodes__)
            associationTypePre = type(self.__associatedMeasurements__)
            self.__terminatedTargets__.append(
                copy.deepcopy(self.__trackNodes__[trackIndex]))
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
            assert nTracksPost == nTracksPre - \
                1, str(nTracksPre) + '=>' + str(nTracksPost)
            assert nAssociationsPost == nAssociationsPre - 1
            assert targetListTypePost == targetListTypePre
            assert trackListTypePost == trackListTypePre
            assert associationTypePost == associationTypePre

    def _processLeafNodes(self, targetNodes, scanList, aisList):
        dummyNodesData = self.__predictDummyMeasurements(targetNodes)

        gatedRadarData = self.__processMeasurements(targetNodes,
                                                    scanList,
                                                    dummyNodesData,
                                                    pv.C_RADAR,
                                                    self.R_RADAR)

        radarNodesData = self.__createPureRadarNodes(gatedRadarData)

        fusedNodesData = self.__fuseRadarAndAis(targetNodes,
                                                aisList,
                                                scanList)

        return dummyNodesData, radarNodesData, fusedNodesData

    @staticmethod
    def __createPureRadarNodes(gatedRadarData):
        (gated_radar_indices_list,
         _,
         gated_x_radar_hat_list,
         P_radar_hat_list,
         _,
         _,
         radar_nllr_list) = gatedRadarData

        newNodesData = (gated_x_radar_hat_list,
                        P_radar_hat_list,
                        gated_radar_indices_list,
                        radar_nllr_list)
        assert all(d is not None for d in newNodesData)
        return newNodesData

    def __fuseRadarAndAis(self, targetNodes, aisList, scanList):
        nNodes = len(targetNodes)

        if aisList is None:
            return ([np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)],
                    [np.array([]) for _ in range(nNodes)])

        aisMeasurements = aisList
        radarMeasurements = scanList.measurements
        aisTimeSet = {m.time for m in aisMeasurements}
        scanTime = scanList.time

        fused_x_hat_list = []
        fused_P_hat_list = []
        fused_radar_indices_list = []
        fused_nllr_list = []
        fused_mmsi_list = []

        lambda_ais = (len(self.__targetList__) * self.P_ais) / (np.pi * self.radarRange ** 2)
        # print("lambda_ais {:.2e}".format(lambda_ais))

        for i, node in enumerate(targetNodes):
            # print("Node",i, "Target ID", node.ID)
            x_hat_list = []
            P_hat_list = []
            radar_indices_list = []
            nllr_list = []
            mmsi_list = []
            for aisTime in aisTimeSet:
                # print("aisTime", aisTime)
                dT1 = float(aisTime) - node.time
                x_bar1, P_bar1 = kalman.predict_single(pv.Phi(dT1), pv.Q(dT1), node.x_0, node.P_0)
                # print("x_bar_1", x_bar1)
                # print("P_bar1\n", P_bar1)
                for highAccuracy in [True, False]:
                    # print("highAccuracy",highAccuracy)
                    z_hat_list1, S_list1, S_inv_list1, K_list1, P_hat_list1 = kalman.precalc(
                        ais_model.C,
                        ais_model.R(highAccuracy),
                        np.array(x_bar1, ndmin=2),
                        np.array(P_bar1, ndmin=3))
                    activeAisMeasurements = [m for m in aisMeasurements if
                                             m.time == aisTime and m.highAccuracy == highAccuracy]
                    if len(activeAisMeasurements) == 0: continue
                    # print("activeAisMeasurements",activeAisMeasurements)
                    # activeAisMeasurementsIndices = np.array([aisMeasurements.index(m) for m in activeAisMeasurements])
                    # print("activeAisMeasurementsIndices",activeAisMeasurementsIndices)
                    z_array1 = np.array([m.state for m in activeAisMeasurements], ndmin=2)
                    z_tilde_array1 = z_array1 - z_hat_list1[0]
                    nis_array1 = (kalman.normalizedInnovationSquared(z_tilde_array1, S_inv_list1))[0]
                    # print("nis_array1",nis_array1)
                    gated_nis_array1 = nis_array1 <= self.eta2_ais
                    # print("gated nis array1", nis_array1[gated_nis_array1])
                    gated_ais_indices = np.flatnonzero(gated_nis_array1)
                    if len(gated_ais_indices) == 0: continue
                    # print("gated S array1\n", S_list1)
                    # print("gated_ais_indices",gated_ais_indices)
                    nllr1_list = kalman.nllr(lambda_ais, 1.0, S_list1, nis_array1[gated_ais_indices])
                    # print("nllr1", np.array_str(nllr1_list, precision=2))
                    for i, ais_index in enumerate(gated_ais_indices):
                        # print("ais_index",ais_index)
                        ais_measurement = activeAisMeasurements[ais_index]
                        x_hat1 = x_bar1 + K_list1[0].dot(ais_measurement.state - z_hat_list1[0])
                        P_hat1 = P_hat_list1[0]
                        dT2 = scanTime - float(ais_measurement.time)
                        x_bar2, P_bar2 = kalman.predict_single(pv.Phi(dT2), pv.Q(dT2), x_hat1, P_hat1)
                        z_hat_list2, S_list2, S_inv_list2, K_list2, P_hat_list2 = kalman.precalc(
                            pv.C_RADAR,
                            pv.R_RADAR(),
                            np.array(x_bar2, ndmin=2),
                            np.array(P_bar2, ndmin=3))
                        z_tilde_array2 = radarMeasurements - z_hat_list2[0]
                        nis_array2 = (kalman.normalizedInnovationSquared(z_tilde_array2, S_inv_list2))[0]
                        gated_nis_array2 = nis_array2 <= self.eta2
                        gated_radar_indices = np.flatnonzero(gated_nis_array2)
                        nllr2_list = kalman.nllr(self.lambda_ex, node.P_d, S_list2, nis_array2[gated_radar_indices])
                        # print("nllr2_list",np.array_str(nllr2_list, precision=2))
                        for j, radar_index in enumerate(gated_radar_indices):
                            x_hat2 = x_bar2 + K_list2[0].dot(radarMeasurements[radar_index] - z_hat_list2[0])
                            P_hat2 = P_hat_list2[0]
                            nllr12 = 0.5 * nllr1_list[i] + 0.5 * nllr2_list[j]
                            log.debug("Fused node " +
                                      np.array_str(x_hat2, precision=1) + " " +
                                      '{: .2f} '.format(nllr12) +
                                      str(ais_measurement.mmsi))

                            x_hat_list.append(x_hat2)
                            P_hat_list.append(P_hat2)
                            radar_indices_list.append(radar_index)
                            nllr_list.append(nllr12)
                            mmsi_list.append(ais_measurement.mmsi)
                        if len(gated_radar_indices) == 0:
                            x_hat2 = x_bar2
                            P_hat2 = P_hat_list2[0]
                            nllr12 = nllr1_list[i]
                            log.debug("Pure AIS node " +
                                      np.array_str(x_hat2, precision=1) + " " +
                                      '{: .2f} '.format(nllr12) +
                                      'MMSI: {:}'.format(ais_measurement.mmsi))
                            x_hat_list.append(x_hat2)
                            P_hat_list.append(P_hat2)
                            radar_indices_list.append(None)
                            nllr_list.append(nllr12)
                            mmsi_list.append(ais_measurement.mmsi)

            fused_x_hat_list.append(np.array(x_hat_list, ndmin=2))
            fused_P_hat_list.append(np.array(P_hat_list, ndmin=3))
            fused_radar_indices_list.append(np.array(radar_indices_list))
            fused_nllr_list.append(np.array(nllr_list))
            fused_mmsi_list.append(np.array(mmsi_list))

        assert len(fused_x_hat_list) == nNodes
        assert len(fused_P_hat_list) == nNodes
        assert len(fused_radar_indices_list) == nNodes
        assert len(fused_nllr_list) == nNodes
        assert len(fused_mmsi_list) == nNodes
        for i in range(nNodes):
            assert fused_x_hat_list[i].ndim == 2, str(fused_x_hat_list[i].ndim)
            assert fused_P_hat_list[i].ndim == 3, str(fused_P_hat_list[i].ndim)
            nFusedNodes, nStates = fused_x_hat_list[i].shape
            if nStates == 0: continue
            assert fused_P_hat_list[i].shape == (nFusedNodes, nStates, nStates), str(fused_P_hat_list[i].shape)

        fusedNodesData = (fused_x_hat_list,
                          fused_P_hat_list,
                          fused_radar_indices_list,
                          fused_nllr_list,
                          fused_mmsi_list)

        return fusedNodesData

    def __fuseRadarAndAisBulk(self, targetNodes, aisList, scanList):
        # TODO: Remove when done debugging
        np.set_printoptions(precision=1, suppress=True)

        nNodes = len(targetNodes)

        if aisList is None:
            return ([None]*nNodes,
                    [None]*nNodes,
                    [None]*nNodes,
                    [None]*nNodes,
                    [None]*nNodes)

        nNodes = len(targetNodes)
        aisMeasurements = aisList
        nAisMeasurements = len(aisMeasurements)
        radarMeasurements = scanList.measurements
        nRadarMeasurements = len(radarMeasurements)
        aisTimeSet = {m.time for m in aisMeasurements}
        scanTime = scanList.time
        nodesTime = scanTime - self.radarPeriod
        ais_meas_dim = ais_model.C.shape[0]
        radar_meas_dim = pv.C_RADAR.shape[0]
        state_dim = ais_model.C.shape[1]

        print("\nnNodes", nNodes)
        print("nAisMeasurements", nAisMeasurements)

        fused_x_hat_list = [None]*nNodes
        fused_P_hat_list = [None]*nNodes
        fused_radar_indices_list = [None]*nNodes
        fused_nllr_list = [None]*nNodes
        fused_mmsi_list = [None]*nNodes

        print("fused_x_hat_list",fused_x_hat_list)

        lambda_ais = (len(self.__targetList__) * self.P_ais) / (np.pi * self.radarRange**2)
        # print("lambda_ais {:.2e}".format(lambda_ais))

        for aisTime in aisTimeSet:
            print("aisTime", aisTime)
            dT1 = float(aisTime) - nodesTime
            assert dT1 >= 0, str(dT1)

            x_0_list = np.array([node.x_0 for node in targetNodes], ndmin=2, dtype=np.float32)
            assert len(x_0_list) == nNodes, str(len(x_0_list))
            P_0_list = np.array([node.P_0 for node in targetNodes], ndmin=3, dtype=np.float32)
            assert len(P_0_list) == nNodes, str(len(P_0_list))
            x_bar1, P_bar1 = kalman.predict(pv.Phi(dT1), pv.Q(dT1), x_0_list, P_0_list)
            print("x_bar1", np.array_str(x_bar1, precision=1, suppress_small=True))
            assert len(x_bar1) == nNodes, str(len(x_bar1))
            assert len(P_bar1) == nNodes, str(len(P_bar1))

            for highAccuracy in [True, False]:
                print("highAccuracy", highAccuracy)
                activeAisMeasurements = [m for m in aisMeasurements
                                         if m.time == aisTime and
                                         m.highAccuracy == highAccuracy]
                nActiveAisMeasurements = len(activeAisMeasurements)
                print("nActiveAisMeasurements", nActiveAisMeasurements)
                if nActiveAisMeasurements == 0:
                    print("No active measurements. Skipping")
                    continue

                z_hat_list1, S_list1, S_inv_list1, K_list1, P_hat_list1 = kalman.precalc(
                    ais_model.C,
                    ais_model.R(highAccuracy),
                    np.array(x_bar1, ndmin=2),
                    np.array(P_bar1, ndmin=3))
                assert z_hat_list1.shape == (nNodes, ais_meas_dim)
                assert S_list1.shape == (nNodes, ais_meas_dim, ais_meas_dim)
                assert S_inv_list1.shape == (nNodes, ais_meas_dim, ais_meas_dim)
                assert K_list1.shape == (nNodes, state_dim, ais_meas_dim)
                assert P_hat_list1.shape == (nNodes, state_dim, state_dim)

                mmsi_array = np.array([m.mmsi for m in activeAisMeasurements], dtype=np.int)
                print("K_list1\n", K_list1)

                z_array1 = np.array([m.state for m in activeAisMeasurements], ndmin=2)
                assert len(z_array1) == nActiveAisMeasurements

                z_tilde_array1 = kalman.z_tilde(z_array1, z_hat_list1, nNodes, ais_meas_dim)
                assert z_tilde_array1.shape == (nNodes, nActiveAisMeasurements, ais_meas_dim)
                print("z_tilde_array1", z_tilde_array1.shape, "\n", z_tilde_array1)

                nis_array1 = np.array((kalman.normalizedInnovationSquared(z_tilde_array1, S_inv_list1)), ndmin=2)
                assert nis_array1.shape == (nNodes, nActiveAisMeasurements)
                print("nis_array1\n", np.array_str(nis_array1, precision=1))

                gate_nis_array1 = np.array(nis_array1 <= self.eta2_ais, dtype=np.bool, ndmin=2)
                assert gate_nis_array1.shape == (nNodes, nActiveAisMeasurements)
                print("gate_nis_array1\n", gate_nis_array1)

                gated_ais_indices = np.nonzero(gate_nis_array1)
                assert len(gated_ais_indices) == 2
                assert len(gated_ais_indices[0]) == len(gated_ais_indices[1])

                nUniqueGatedAisMeasurements = len(set(gated_ais_indices[1]))
                nGatedAisMeasurements = np.count_nonzero(gate_nis_array1, axis=1)
                if nUniqueGatedAisMeasurements == 0:
                    print("No AIS measurements inside the gate. Skipping.")
                    continue
                print("nGatedAisMeasurements", nGatedAisMeasurements)
                print("nUniqueGatedAisMeasurements", nUniqueGatedAisMeasurements)
                print("Node indices", gated_ais_indices[0],
                      "are connected with AIS indices", gated_ais_indices[1], "respectively")

                gated_nis1 = [np.array([nis_array1[i1][i2]], ndmin=1)
                                       for i1, i2 in zip(gated_ais_indices[0], gated_ais_indices[1])]
                # gated_nis1 = np.array(nis_array1[gated_ais_indices], ndmin=2)
                print("gated_nis1\n", gated_nis1)
                assert len(gated_nis1) == len(gated_ais_indices[0])
                # assert all([gated_nis1[i].shape == (nGatedAisMeasurements[i],1) for i,e in enumerate(nGatedAisMeasurements) ])

                nllr1_list = [kalman.nllr(lambda_ais, 1.0, S_list1[i], gated_nis1[i]) for i in range(len(gated_nis1))]
                print("nllr1_list", nllr1_list)
                # assert nllr1_list.shape == (nNodes, nGatedAisMeasurements), str(nllr1_list.shape)

                print("x_bar1[gated_ais_indices[0]]", x_bar1[gated_ais_indices[0]])
                print("K_list1[gated_ais_indices[0]]\n", K_list1[gated_ais_indices[0]])
                print("z_tilde_array1[gated_ais_indices]\n", z_tilde_array1[gated_ais_indices])

                x_hat1 = (x_bar1[gated_ais_indices[0]] + np.matmul(K_list1[gated_ais_indices[0]],
                                                                   z_tilde_array1[gated_ais_indices].T)[0].T)
                print("x_hat1", x_hat1.shape, "\n", x_hat1)
                assert x_hat1.shape == (np.sum(nGatedAisMeasurements), state_dim), str(x_hat1.shape)

                P_hat1 = P_hat_list1[gated_ais_indices[0]]
                print("P_hat1", P_hat1.shape, "\n", P_hat1)
                assert P_hat1.shape == (x_hat1.shape[0], state_dim, state_dim)

                dT2 = scanTime - float(aisTime)
                assert dT2 >= 0, str(dT2)

                x_bar2, P_bar2 = kalman.predict(pv.Phi(dT2), pv.Q(dT2), x_hat1, P_hat1)
                assert x_bar2.shape == x_hat1.shape
                assert P_bar2.shape == P_hat1.shape

                z_hat_list2, S_list2, S_inv_list2, K_list2, P_hat_list2 = kalman.precalc(
                    pv.C_RADAR,
                    pv.R_RADAR(),
                    np.array(x_bar2, ndmin=2),
                    np.array(P_bar2, ndmin=3))
                print("z_hat_list2\n", z_hat_list2)
                # print("K_list2\n", K_list2)
                assert z_hat_list2.shape == (np.sum(nGatedAisMeasurements), radar_meas_dim)
                assert S_list2.shape == (np.sum(nGatedAisMeasurements), radar_meas_dim, radar_meas_dim)
                assert S_inv_list2.shape == (np.sum(nGatedAisMeasurements), radar_meas_dim, radar_meas_dim)
                assert K_list2.shape == (np.sum(nGatedAisMeasurements), state_dim, radar_meas_dim)
                assert P_hat_list2.shape == (np.sum(nGatedAisMeasurements), state_dim, state_dim)

                print("nRadarMeasurements", nRadarMeasurements)

                z_tilde_array2 = kalman.z_tilde(radarMeasurements, z_hat_list2, np.sum(nGatedAisMeasurements), radar_meas_dim)
                print("z_tilde_array2", z_tilde_array2.shape, "\n", z_tilde_array2)
                assert z_tilde_array2.shape == (np.sum(nGatedAisMeasurements), nRadarMeasurements, radar_meas_dim)

                nis_array2 = (kalman.normalizedInnovationSquared(z_tilde_array2, S_inv_list2))
                # print("nis_array2", nis_array2.shape, "\n", nis_array2)
                assert nis_array2.shape == (np.sum(nGatedAisMeasurements), nRadarMeasurements)

                gated_nis_array2 = nis_array2 <= self.eta2
                # print("gated_nis_array2", gated_nis_array2.shape, "\n", np.asarray(gated_nis_array2, dtype=np.int))
                assert gated_nis_array2.shape == (np.sum(nGatedAisMeasurements), nRadarMeasurements)

                gated_radar_indices = np.nonzero(gated_nis_array2)
                assert len(gated_radar_indices[0]) == len(gated_radar_indices[1])
                print("gated_radar_indices", gated_radar_indices)
                nGatedRadarMeasurements = len(gated_radar_indices[0])
                print("nGatedRadarMeasurements", nGatedRadarMeasurements)
                print("AIS indices", gated_radar_indices[0],
                      "(",gated_ais_indices[1][gated_radar_indices[0]],")",
                      "[",gated_ais_indices[0][gated_radar_indices[0]],"]",
                        "are connected with radar indices", gated_radar_indices[1], "respectively")
                nllr2_list = kalman.nllr(self.lambda_ex, self.default_P_d, S_list2, nis_array2[gated_radar_indices])
                print("nllr2_list", nllr2_list)
                assert nllr2_list.ndim == 1
                assert nllr2_list.size == nGatedRadarMeasurements

                if nGatedRadarMeasurements > 0:
                    x_hat2 = (x_bar2[gated_radar_indices[0]] +
                              np.matmul(K_list2[gated_radar_indices[0]],
                                        z_tilde_array2[gated_radar_indices].T)[0].T)
                    print("x_hat2", x_hat2.shape, "\n", x_hat2)
                    assert x_hat2.shape == (nGatedRadarMeasurements, state_dim)
                    P_hat2 = P_hat_list2[gated_radar_indices[0]]
                    assert P_hat2.shape == (x_hat2.shape[0], state_dim, state_dim)
                    # print("P_hat2", P_hat2.shape, "\n", P_hat2)

                    fused_node_indices = [np.where(gated_ais_indices[0]==i for i in gated_radar_indices[0])]
                    print("fused_node_indices",fused_node_indices)



                    continue
                for nodeIndex in range(nNodes):
                    # nodeAisIndices = np.where(gated_ais_indices[0]==nodeIndex)[0]
                    # if len(nodeAisIndices) == 0:
                    #     continue
                    # print("nodeAisIndices",nodeAisIndices)
                    # nodeGatedAisIndices = gated_ais_indices[1][nodeAisIndices]
                    # print("nodeGatedAisIndices",nodeGatedAisIndices)
                    if nGatedRadarMeasurements > 0:
                        for nodeIndex in gated_ais_indices[0]:
                            fused_x_hat_list[nodeIndex] = x_hat2
                            fused_P_hat_list[nodeIndex] = P_hat2
                            fused_radar_indices_list[nodeIndex] = gated_radar_indices[1]
                            fused_nllr_list[nodeIndex] = nllr2_list
                            fused_mmsi_list[nodeIndex] = mmsi_array[gated_radar_indices[0]]
                    else:
                        print("No radar measurement inside AIS gates. Creating pure AIS node")
                        # log.debug("Pure AIS node " +
                        #           np.array_str(x_hat2, precision=1) + " " +
                        #           '{: .2f} '.format(nllr12) +
                        #           'MMSI: {:}'.format(ais_measurement.mmsi))
                        for nodeIndex, aisIndex in zip(gated_ais_indices[0], gated_ais_indices[1]):
                            x_hat2 = x_bar2
                            P_hat2 = P_bar2
                            fused_x_hat_list[nodeIndex] = x_hat2
                            fused_P_hat_list[nodeIndex] = P_hat2
                            fused_radar_indices_list[nodeIndex] = None
                            fused_nllr_list[nodeIndex] = nllr1_list
                            fused_mmsi_list[nodeIndex] = mmsi_array[aisIndex]

        print("fused_x_hat_list", fused_x_hat_list)
        print("fused_P_hat_list", fused_P_hat_list)
        print("fused_radar_indices_list", fused_radar_indices_list)
        print("fused_nllr_list", fused_nllr_list)
        print("fused_mmsi_list", fused_mmsi_list)

        assert len(fused_x_hat_list) == nNodes
        assert len(fused_P_hat_list) == nNodes
        assert len(fused_radar_indices_list) == nNodes
        assert len(fused_nllr_list) == nNodes
        assert len(fused_mmsi_list) == nNodes
        # for i in range(nNodes):
        #     assert fused_x_hat_list[i].ndim == 2, str(fused_x_hat_list[i].ndim)
        #     assert fused_P_hat_list[i].ndim == 3, str(fused_P_hat_list[i].ndim)
        #     nFusedNodes, nStates = fused_x_hat_list[i].shape
        #     if nStates == 0:
        #         continue
        #     assert fused_P_hat_list[i].shape == (
        #         nFusedNodes, nStates, nStates), str(fused_P_hat_list[i].shape)

        fusedNodesData = (fused_x_hat_list,
                          fused_P_hat_list,
                          fused_radar_indices_list,
                          fused_nllr_list,
                          fused_mmsi_list)

        return fusedNodesData

    def __processMeasurements(self, targetNodes, measurementList, dummyNodesData, C, R):
        if measurementList is None:
            return None
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
                np.array(nis[gated_filter], ndmin=2),
                nllr_list)

    def __predictDummyMeasurements(self, targetNodes):
        nNodes = len(targetNodes)
        radarMeasDim, nStates = pv.C_RADAR.shape
        x_0_list = np.array([target.x_0 for target in targetNodes],
                            ndmin=2)
        P_0_list = np.array([target.P_0 for target in targetNodes],
                            ndmin=3)
        assert x_0_list.shape == (nNodes, nStates)
        assert P_0_list.shape == (nNodes, nStates, nStates)

        x_bar_list, P_bar_list = kalman.predict(
            self.A, self.Q, x_0_list, P_0_list)
        return x_bar_list, P_bar_list

    def __predictPrecalcBulk(self, targetNodes, C, R, dummyNodesData):
        nNodes = len(targetNodes)
        measDim, nStates = C.shape
        x_bar_list, P_bar_list = dummyNodesData

        z_hat_list, S_list, S_inv_list, K_list, P_hat_list = kalman.precalc(
            C, R, x_bar_list, P_bar_list)

        assert S_list.shape == (nNodes, measDim, measDim)
        assert S_inv_list.shape == (nNodes, measDim, measDim)
        assert K_list.shape == (nNodes, nStates, measDim)
        assert P_hat_list.shape == P_bar_list.shape
        assert z_hat_list.shape == (nNodes, measDim)

        return z_hat_list, S_list, S_inv_list, K_list, P_hat_list

    def __analyzeTrackTermination(self):
        deadTracks = []
        for trackIndex, trackNode in enumerate(self.__trackNodes__):
            # Check outside radarRange
            if trackNode.isOutsideRange(self.position, self.radarRange):
                trackNode.status = outofrangeTag
                deadTracks.append(trackIndex)
                log.info("Terminating track {0:} at {1:} since it is out of radarRange".format(
                    trackIndex, np.array_str(self.__trackNodes__[trackIndex].x_0[0:2])))

            # Check if track is to insecure
            elif trackNode.getScore() / (self.N+1) > self.scoreUpperLimit:
                trackNode.status = toolowscoreTag
                deadTracks.append(trackIndex)
                log.info("Terminating track {0:} at {1:} since its score is above the threshold ({2:.1f}>{3:.1f})".format(
                    trackIndex, np.array_str(self.__trackNodes__[trackIndex].x_0[0:2]),
                    trackNode.getScore() / (self.N+1), self.scoreUpperLimit))
            elif trackNode.cumulativeNLLR > self.clnnrUpperLimit:
                trackNode.status = toolowscoreTag
                deadTracks.append(trackIndex)
                log.info(
                    "Terminating track {0:} at {1:} since its CNNLR is above the threshold ({2:.1f}>{3:.1f})".format(
                        trackIndex, np.array_str(
                            self.__trackNodes__[trackIndex].x_0[0:2]),
                        trackNode.cumulativeNLLR, self.clnnrUpperLimit))
        return deadTracks

    def __dynamicWindow(self, targetProcessTimes):
        totalGrowTime = sum(targetProcessTimes)
        tooSlowTotal = totalGrowTime > self.totalGrowTimeLimit
        targetProcessTimeLimit = (self.totalGrowTimeLimit / len(self.__targetList__)
                                  if tooSlowTotal else self.nodeGrowTimeLimit)
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
        if tempTotalTime > (self.radarPeriod * 0.8):
            self.N = max(1, self.N - 1)
            log.warning(
                'Iteration took to long time ({0:.1f}ms), reducing window size roof from {1:} to  {2:}'.format(
                    tempTotalTime * 1000, self.N + 1, self.N))
            self.__targetWindowSize__ = [min(e, self.N)
                                         for e in self.__targetWindowSize__]

    def _compareTracksWithTruth(self, xTrue):
        return [(target.filteredStateMean - xTrue[targetIndex].state).T.dot(
            np.linalg.inv(target.filteredStateCovariance)).dot(
            (target.filteredStateMean - xTrue[targetIndex].state))
            for targetIndex, target in enumerate(self.__trackNodes__)]

    def getRuntimeAverage(self):
        return {k: np.mean(np.array(v)) for k, v in self.runtimeLog.items()}

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
        log.debug("Cluster {0:} Sum = {1:}".format(cluster, len(cluster)))
        nHypInClusterArray = self._getHypInCluster(cluster)
        log.debug("nHypInClusterArray {0:} => Sum = {1:}".format(
            nHypInClusterArray, sum(nHypInClusterArray)))

        for i in cluster:
            log.debug("AssociatedMeasurements[{0:}] {1:}".format(
                i, self.__associatedMeasurements__[i]))
        uniqueMeasurementSet = set.union(
            *[self.__associatedMeasurements__[i] for i in cluster])
        nRealMeasurementsInCluster = len(uniqueMeasurementSet)
        log.debug("Cluster Measurement set: {0:} Sum={1:}".format(
            uniqueMeasurementSet, nRealMeasurementsInCluster))

        (A1, measurementList) = self._createA1(
            nRealMeasurementsInCluster, sum(nHypInClusterArray), cluster)
        log.debug("Difference: {:}".format(
            uniqueMeasurementSet.symmetric_difference(set(measurementList))))

        assert len(measurementList) == A1.shape[0], str(
            len(measurementList)) + " vs " + str(A1.shape[0])

        assert len(measurementList) == nRealMeasurementsInCluster
        A2 = self._createA2(len(cluster), nHypInClusterArray)
        log.debug("A2 \n" + np.array_str(A2.astype(np.int), max_line_width=200))
        C = self._createC(cluster)
        log.debug("C =" + np.array_str(np.array(C), precision=1))

        log.debug("Solving optimal association in cluster with targets" +
                  str(cluster) + ",   \t" +
                  str(sum(nHypInClusterArray)) + " hypotheses and " +
                  str(nRealMeasurementsInCluster) + " real measurements.")
        selectedHypotheses = self._solveBLP_OR_TOOLS(A1, A2, C)
        log.debug("selectedHypotheses" + str(selectedHypotheses))
        selectedNodes = self._hypotheses2Nodes(selectedHypotheses, cluster)
        selectedNodesArray = np.array(selectedNodes)

        assert len(selectedHypotheses) == len(cluster), \
            "__solveOptimumAssociation did not find the correct number of hypotheses"
        assert len(selectedNodes) == len(cluster), \
            "did not find the correct number of nodes"
        assert len(selectedHypotheses) == len(set(selectedHypotheses)), \
            "selected two or more equal hypotheses"
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
            if target.trackHypotheses is None:  # leaf node

                if ((target.measurementNumber is not None) and
                        (target.measurementNumber != 0)):  # we are at a real measurement
                    radarMeasurement = (target.scanNumber, target.measurementNumber)
                    try:
                        radarMeasurementIndex = measurementList.index(radarMeasurement)
                    except ValueError:
                        measurementList.append(radarMeasurement)
                        radarMeasurementIndex = len(measurementList) - 1
                    activeMeasurements[radarMeasurementIndex] = True

                if target.mmsi is not None:
                    aisMeasurement = (target.scanNumber, target.mmsi)
                    try:
                        aisMeasurementIndex = measurementList.index(aisMeasurement)
                    except ValueError:
                        measurementList.append(aisMeasurement)
                        aisMeasurementIndex = len(measurementList) - 1
                    activeMeasurements[aisMeasurementIndex] = True

                A1[activeMeasurements, hypothesisIndex[0]] = True
                hypothesisIndex[0] += 1

            else:
                for hyp in target.trackHypotheses:
                    activeMeasurementsCpy = activeMeasurements.copy()
                    if ((hyp.measurementNumber is not None) and
                            (hyp.measurementNumber != 0)):
                        radarMeasurement = (hyp.scanNumber, hyp.measurementNumber)
                        try:
                            radarMeasurementIndex = measurementList.index(
                                radarMeasurement)
                        except ValueError:
                            measurementList.append(radarMeasurement)
                            radarMeasurementIndex = len(measurementList) - 1
                        activeMeasurementsCpy[radarMeasurementIndex] = True

                    if hyp.mmsi is not None:
                        aisMeasurement = (hyp.scanNumber, hyp.mmsi)
                        try:
                            aisMeasurementIndex = measurementList.index(aisMeasurement)
                        except ValueError:
                            measurementList.append(aisMeasurement)
                            aisMeasurementIndex = len(measurementList) - 1
                        activeMeasurementsCpy[aisMeasurementIndex] = True

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
        log.debug("measurementList" + str(measurementList) +
                  "Sum=" + str(len(measurementList)))
        log.debug("size(A1) " + str(A1.shape))
        log.debug("A1 \n" + 'V: Measurements, LeafNodes ---->\n' +
                  np.array_str(A1.astype(np.int), max_line_width=200))
        # assert len(measurementList) == A1.shape[0], str(len(measurementList)) + " vs " + str(A1.shape[0])
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
                scoreArray.append(target.getScore() / self.N)
            else:
                for hyp in target.trackHypotheses:
                    getTargetScore(hyp, scoreArray)

        scoreArray = []
        for targetIndex in cluster:
            getTargetScore(self.__targetList__[targetIndex], scoreArray)
        assert all(np.isfinite(scoreArray)), str(scoreArray)
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

    def _solveBLP_OR_TOOLS(self, A1, A2, f):

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
        log.debug("Optim Time = " + str(solver.WallTime()) + " milliseconds")

        if result_status == pywraplp.Solver.OPTIMAL:
            log.debug("Optim result optimal")
        else:
            log.warning("Optim result NOT optimal")

        toc2 = time.time() - tic2

        tic3 = time.time()
        selectedHypotheses = [i for i in range(nHyp)
                              if tau[i].solution_value() > 0.]
        log.debug("Selected hypotheses" + str(selectedHypotheses))
        assert len(selectedHypotheses) == nTargets
        toc3 = time.time() - tic3

        log.debug('_solveBLP_OR_TOOLS ({0:4.0f}|{1:4.0f}|{2:4.0f}|{3:4.0f}) ms = {4:4.0f}'.format(
            toc0 * 1000, toc1 * 1000, toc2 * 1000, toc3 * 1000, (toc0 + toc1 + toc2 + toc3) * 1000))
        return selectedHypotheses

    def _pruneTargetIndex(self, targetIndex, N):
        node = self.__trackNodes__[targetIndex]
        newRootNode = node.pruneDepth(N)
        if newRootNode != self.__targetList__[targetIndex]:
            newRootNode.parent.isRoot = False
            newRootNode.isRoot = True
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
            self.__associatedMeasurements__[targetIndex] = self.__targetList__[
                targetIndex].getMeasurementSet()

    def _checkTrackerIntegrity(self):
        log.debug("Checking tracker integrity")
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
                assert leafNode.scanNumber == scanNumber, \
                    "{0:} != {1:} @ TargetNumber {2:}".format(leafNode.scanNumber,
                                                              scanNumber,
                                                              targetIndex + 1)
                leafNode._checkMmsiIntegrity()
                assert np.isfinite(leafNode.getScore())
                assert np.isfinite(leafNode.cumulativeNLLR)
        activeMmsiList = [target.mmsi
                          for target in self.__trackNodes__
                          if target.mmsi is not None]
        activeMmsiSet = set(activeMmsiList)
        assert len(activeMmsiList) == len(
            activeMmsiSet), "One or more MMSI is used multiple times"

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

    def plotHypothesesTrack(self, ax=plt.gca(), **kwargs):
        def recPlotHypothesesTrack(ax, target, track=[], **kwargs):
            newTrack = track[:] + [target.getPosition()]
            if target.trackHypotheses is None:
                ax.plot([p.x() for p in newTrack],
                         [p.y() for p in newTrack],
                         "--",
                         **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    recPlotHypothesesTrack(ax, hyp, newTrack, **kwargs)

        colors = kwargs.get("colors", self._getColorCycle())
        for target in self.__targetList__:
            recPlotHypothesesTrack(ax, target, c=next(colors))
        if kwargs.get('markStates', False):
            defaults = {'dummy': True, 'real': True, 'ais': True,
                        'includeHistory': False, 'color': 'red'}
            self.plotStatesFromRoot(ax, **{**defaults, **kwargs})

    def plotActiveTracks(self, ax=plt.gca(), **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for i, track in enumerate(self.__trackNodes__):
            track.plotTrack(ax, root=self.__targetList__[i], c=next(
                colors), period=self.radarPeriod, **kwargs)
        if kwargs.get('markStates', True):
            defaults = {'labels': False, 'dummy': True,
                        'real': True, 'ais': True, 'color': 'red'}
            self.plotStatesFromTracks(ax, **{**defaults, **kwargs})

    def plotTerminatedTracks(self, ax=plt.gca(), **kwargs):
        colors = kwargs.get("colors", self._getColorCycle())
        for track in self.__terminatedTargets__:
            defaults = {'c': next(colors), 'markInitial': True,
                        'markEnd': True, 'terminated': True}
            track.plotTrack(ax, **{**defaults, **kwargs})
            if kwargs.get('markStates', False):
                defaults = {'labels': False, 'dummy': True, 'real': True, 'ais': True}
                track.plotStates(ax, float('inf'), **{**defaults, **kwargs})

    def plotMeasurementsFromTracks(self, ax=plt.gca(), stepsBack=float('inf'), **kwargs):
        for node in self.__trackNodes__:
            node.plotMeasurement(ax, stepsBack, **kwargs)

    def plotStatesFromTracks(self, ax=plt.gca(), stepsBack=float('inf'), **kwargs):
        for node in self.__trackNodes__:
            node.plotStates(ax, stepsBack, **kwargs)

    def plotMeasurementsFromRoot(self, ax=plt.gca(), **kwargs):
        if not (("real" in kwargs) or ("dummy" in kwargs) or ("ais" in kwargs)):
            return
        plottedMeasurements = set()
        for target in self.__targetList__:
            if kwargs.get("includeHistory", False):
                target.getInitial().recDownPlotMeasurements(plottedMeasurements, ax, **kwargs)
            else:
                for hyp in target.trackHypotheses:
                    hyp.recDownPlotMeasurements(plottedMeasurements, ax, **kwargs)

    def plotStatesFromRoot(self, ax=plt.gca(), **kwargs):
        if not (("real" in kwargs) or ("dummy" in kwargs) or ("ais" in kwargs)):
            return
        for target in self.__targetList__:
            if kwargs.get("includeHistory", False):
                target.getInitial().recDownPlotStates(ax, **kwargs)
            elif target.trackHypotheses is not None:
                for hyp in target.trackHypotheses:
                    hyp.recDownPlotStates(ax, **kwargs)

    def plotScanIndex(self, index, ax=plt.gca(), **kwargs):
        self.__scanHistory__[index].plot(ax, **kwargs)

    def plotLastScan(self, ax=plt.gca(), **kwargs):
        self.__scanHistory__[-1].plot(ax, **kwargs)

    def plotLastAisUpdate(self, ax=plt.gca(), **kwargs):
        if self.__aisHistory__[-1] is not None:
            self.__aisHistory__[-1].plot(ax, **kwargs)

    def plotAllScans(self, ax=plt.gca(), stepsBack=None, **kwargs):
        if stepsBack is not None:
            stepsBack = -(stepsBack + 1)
        for scan in self.__scanHistory__[:stepsBack:-1]:
            scan.plot(ax, **kwargs)

    def plotAllAisUpdates(self, ax=plt.gca(), stepsBack=None, **kwargs):
        if stepsBack is not None:
            stepsBack = -(stepsBack + 1)
        for update in self.__aisHistory__[:stepsBack:-1]:
            if update is not None:
                update.plot(ax, markeredgewidth=2, **kwargs)

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

    def getTimeLogHeader(self):
        return ('{:3} '.format("Nr") +
                '{:9} '.format("Num Targets") +
                '{:12} '.format("Iteration (ms)") +
                '({0:23} '.format("(nMeasurements + nAisUpdates / nNodes) Process time ms") +
                '({0:2}) {1:5}'.format("nClusters", 'Cluster') +
                '({0:3}) {1:6}'.format("nOptimSolved", 'Optim') +
                '{:4}'.format('DynN') +
                '{:5}'.format('N-Prune') +
                '{:3}'.format('Terminate') +
                '{:5}'.format('Init')
                )

    def getTimeLogString(self):
        tocMS = {k: v * 1000 for k, v in self.toc.items()}
        totalTime = tocMS['Total']
        nNodes = sum([target.getNumOfNodes() for target in self.__targetList__])
        nMeasurements = len(self.__scanHistory__[-1].measurements)
        nAisUpdates = len(
            self.__aisHistory__[-1]) if self.__aisHistory__[-1] is not None else 0
        scanNumber = len(self.__scanHistory__)
        nTargets = len(self.__targetList__)
        nClusters = len(self.__clusterList__)
        timeLogString = ('{:<3.0f} '.format(scanNumber) +
                         'nTrack {:2.0f} '.format(nTargets) +
                         'Total {0:6.0f} '.format(totalTime) +
                         'Process({0:4.0f}+{1:<3.0f}/{2:6.0f}) {3:6.1f} '.format(
                             nMeasurements, nAisUpdates, nNodes, tocMS['Process']) +
                         'Cluster({0:2.0f}) {1:5.1f} '.format(nClusters, tocMS['Cluster']) +
                         'Optim({0:g}) {1:6.1f} '.format(self.nOptimSolved, tocMS['Optim']) +
                         # 'ILP-Prune {:5.0f}'.format(self.toc['ILP-Prune']) + "  " +
                         'DynN {:4.1f} '.format(tocMS['DynN']) +
                         'N-Prune {:5.1f} '.format(tocMS['N-Prune']) +
                         'Kill {:3.1f} '.format(tocMS['Terminate']) +
                         'Init {:5.1f}'.format(tocMS['Init']))
        return timeLogString

    def printTimeLog(self, **kwargs):
        tooLongWarning = self.toc['Total'] > self.radarPeriod * 0.6
        tooLongCritical = self.toc['Total'] > self.radarPeriod
        on_color = 'on_green'
        on_color = 'on_yellow' if tooLongWarning else on_color
        on_color = 'on_red' if tooLongCritical else on_color
        on_color = kwargs.get('on_color', on_color)
        attrs = ['dark']
        attrs = attrs.append('bold') if tooLongWarning else attrs
        cprint(self.getTimeLogString(),
               on_color=on_color,
               attrs=attrs
               )

    def printTimeLogHeader(self):
        print(self.getTimeLogHeader())

    def printClusterList(clusterList):
        print("Clusters:")
        for clusterIndex, cluster in enumerate(clusterList):
            print("Cluster ", clusterIndex, " contains target(s):\t", cluster,
                  sep="", end="\n")

    def getScenarioElement(self, **kwargs):
        return ET.Element(scenarioTag)

    def _storeTrackerArgs(self, scenarioElement, **kwargs):
        for k, v in kwargs.items():
            scenarioElement.attrib[str(k)] = str(v)

        trackerSettingElement = ET.SubElement(scenarioElement, trackerSettingsTag)
        ET.SubElement(trackerSettingElement, 'M_required').text = str(self.M_required)
        ET.SubElement(trackerSettingElement, 'N_checks').text = str(self.N_checks)
        ET.SubElement(trackerSettingElement, 'mergeThreshold').text = str(
            self.mergeThreshold)
        ET.SubElement(trackerSettingElement, 'ownPosition').text = str(self.position)
        ET.SubElement(trackerSettingElement, 'radarRange').text = str(self.radarRange)
        ET.SubElement(trackerSettingElement, 'radarPeriod').text = str(self.radarPeriod)
        ET.SubElement(trackerSettingElement, 'lambdaPhi').text = str(self.lambda_phi)
        ET.SubElement(trackerSettingElement, 'lambdaNu').text = str(self.lambda_nu)
        ET.SubElement(trackerSettingElement, 'lambdaEx').text = str(self.lambda_ex)
        ET.SubElement(trackerSettingElement, 'eta2').text = str(self.eta2)
        ET.SubElement(trackerSettingElement, 'N_max').text = str(self.N_max)
        ET.SubElement(trackerSettingElement, 'NLLR_upperLimit').text = str(
            self.scoreUpperLimit)
        ET.SubElement(trackerSettingElement, 'pruneThreshold').text = str(
            self.pruneThreshold)
        ET.SubElement(trackerSettingElement, 'targetSizeLimit').text = str(
            self.targetSizeLimit)
        ET.SubElement(trackerSettingElement, 'maxSpeedMS').text = str(self.maxSpeedMS)

    def _storeRun(self, scenarioElement, preInitialized=True, **kwargs):
        runElement = ET.SubElement(scenarioElement, runTag)

        if iterationTag in kwargs:
            iteration = kwargs.get(iterationTag)
        else:
            iteration = len(scenarioElement.findall(runTag))
        runElement.attrib[iterationTag] = str(iteration)

        if seedTag in kwargs:
            runElement.attrib[seedTag] = str(kwargs.get(seedTag))

        runtimeElement = ET.SubElement(runElement,
                                       runtimeTag,
                                       attrib={descriptionTag: "Per iteration",
                                               precisionTag: str(timeLogPrecision)})
        for k, v in self.runtimeLog.items():
            if not v:
                continue
            array = np.array(v)
            mean = np.mean(array)
            min = np.min(array)
            max = np.max(array)
            meanString = str(round(mean, timeLogPrecision))
            minString = str(round(min, timeLogPrecision))
            maxString = str(round(max, timeLogPrecision))
            ET.SubElement(runtimeElement,
                          str(k),
                          attrib={meanTag: meanString,
                                  minTag: minString,
                                  maxTag: maxString}
                          ).text = np.array_str(array,
                                                precision=timeLogPrecision,
                                                max_line_width=999999)

        for target in self.__trackNodes__:
            if preInitialized:
                target._storeNode(runElement, self.radarPeriod)
            else:
                target._storeNodeSparse(runElement)

        for target in self.__terminatedTargets__:
            if preInitialized:
                target._storeNode(runElement, self.radarPeriod, terminated=True)
            else:
                target._storeNodeSparse(runElement, terminated=True)

if __name__ == '__main__':
    pass
