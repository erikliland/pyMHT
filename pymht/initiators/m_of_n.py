import numpy as np
import pymht.models.pv as pv
from pymht.pyTarget import Target
from munkres import Munkres
from scipy.stats import chi2
import logging
import sys

tracking_parameters = {
    'gate_probability': 0.99,
}
tracking_parameters['gamma'] = chi2(df=2).ppf(tracking_parameters['gate_probability'])

CONFIRMED = 1
PRELIMINARY = 0
DEAD = -1

np.set_printoptions(precision=1, suppress=True)


def _solve_global_nearest_neighbour(delta_matrix, gate_distance=np.Inf, **kwargs):
    try:
        DEBUG = kwargs.get('debug', False)
        # Copy and gating
        if DEBUG: print("delta matrix\n", delta_matrix)
        cost_matrix = np.copy(delta_matrix)
        cost_matrix[cost_matrix > gate_distance] = np.Inf
        if DEBUG: print("cost_matrix\n", cost_matrix)

        # Pre-processing
        valid_matrix = cost_matrix < np.Inf
        if np.all(valid_matrix == False):
            return []
        if DEBUG: print("Valid matrix\n", valid_matrix.astype(int))

        bigM = np.power(10., 1.0 + np.ceil(np.log10(1. + np.sum(cost_matrix[valid_matrix]))))
        cost_matrix[np.logical_not(valid_matrix)] = bigM
        if DEBUG: print("Modified cost matrix\n", cost_matrix)

        validCol = np.any(valid_matrix, axis=0)
        validRow = np.any(valid_matrix, axis=1)
        if DEBUG: print("validCol", validCol)
        if DEBUG: print("validRow", validRow)
        nRows = int(np.sum(validRow))
        nCols = int(np.sum(validCol))
        n = max(nRows, nCols)
        if DEBUG: print("nRows, nCols, n", nRows, nCols, n)

        maxv = 10. * np.max(cost_matrix[valid_matrix])
        if DEBUG: print("maxv", maxv)

        rows = np.arange(nRows)
        cols = np.arange(nCols)
        dMat = np.zeros((n, n)) + maxv
        dMat[np.ix_(rows, cols)] = cost_matrix[np.ix_(validRow, validCol)]
        if DEBUG: print("dMat\n", dMat)

        # Assignment
        m = Munkres()
        preliminary_assignments = m.compute(dMat.tolist())
        if DEBUG: print("preliminary assignments", preliminary_assignments)

        # Post-processing
        rowIdx = np.where(validRow)[0]
        colIdx = np.where(validCol)[0]
        assignments = []
        for preliminary_assignment in preliminary_assignments:
            row = preliminary_assignment[0]
            col = preliminary_assignment[1]
            if (row >= nRows) or (col >= nCols):
                break
            rowI = rowIdx[row]
            colI = colIdx[col]
            if valid_matrix[rowI, colI]:
                assignments.append((rowI, colI))
        if DEBUG: print("assignments", assignments)
        return assignments
    except Exception as e:
        print("#" * 20, "CRASH DEBUG INFO", "#" * 20)
        print("deltaMatrix", delta_matrix.shape, "\n", delta_matrix)
        print("gateDistance", gate_distance)
        print("Valid matrix", valid_matrix.shape, "\n", valid_matrix.astype(int))
        print("validCol", validCol.astype(int))
        print("validRow", validRow.astype(int))
        print("dMat", dMat.shape, "\n", dMat)
        print("preliminary assignments", preliminary_assignments)
        print("rowIdx", rowIdx)
        print("colIdx", colIdx)
        print("assignments", assignments)
        print("#" * 20, "CRASH DEBUG INFO", "#" * 20)
        import time
        time.sleep(0.1)
        raise e


def _initiator_distance(delta_vector, dt, v_max, R):
    movement_scalar = dt * v_max
    d_plus = np.maximum(delta_vector - movement_scalar, np.zeros(2))
    d_minus = np.maximum(-delta_vector - movement_scalar, np.zeros(2))
    d = d_plus + d_minus
    D = np.dot(d.T, np.dot(np.linalg.inv(R + R), d))
    return D


def _merge_targets(targets):
    time = targets[0].time
    scanNumber = None
    x_0 = np.mean(np.array([t.x_0 for t in targets]), axis=0)
    assert x_0.shape == targets[0].x_0.shape
    P_0 = np.mean(np.array([t.P_0 for t in targets]), axis=0)
    assert P_0.shape == targets[0].P_0.shape
    return Target(time, scanNumber, x_0, P_0)


def _merge_similar_targets(initial_targets, threshold):
    if not initial_targets: return initial_targets
    targets = []
    used_targets = set()
    for target_index, target in enumerate(initial_targets):
        if target_index not in used_targets:
            distance_to_targets = np.array([np.linalg.norm(target.x_0[0:2] - t.x_0[0:2]) for t in initial_targets])
            close_targets = distance_to_targets < threshold
            close_targets_indices = np.where(close_targets)[0]
            selected_targets = [initial_targets[i] for i in close_targets_indices if i not in used_targets]
            merged_target = _merge_targets(selected_targets)
            for i in close_targets_indices:
                used_targets.add(i)
            assert type(merged_target) == type(target)
            targets.append(merged_target)
    return targets


class PreliminaryTrack():
    def __init__(self, state, covariance):
        self.estimates = [state]
        self.covariance = covariance
        self.n = 0
        self.m = 0
        self.predicted_state = None
        self.measurement_index = None

    def __repr__(self):
        return "({0:}|{1:})".format(self.m, self.n)

    def predict(self, F, Q):
        self.predicted_state = F.dot(self.estimates[-1])
        self.covariance = F.dot(self.covariance).dot(F.T) + pv.Gamma.dot(Q).dot(pv.Gamma.T)

    def mn_analysis(self, M, N):
        n = self.n
        m = self.m
        if n >= N and m >= M:
            return CONFIRMED
        elif n >= N and m < M:
            return DEAD
        else:
            return PRELIMINARY

    def get_predicted_state_and_clear(self):
        return_value = np.copy(self.predicted_state)
        self.predicted_state = None
        return return_value


class Measurement():
    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp
        # self.covariance = pv.R_RADAR()

    def __repr__(self):
        from time import strftime, gmtime
        meas_str = "Measurement: (%.2f, %.2f)" % (self.value[0], self.value[1])
        time_str = "Time: " + strftime("%H:%M:%S", gmtime(self.timestamp))
        return "{" + meas_str + ", " + time_str + "}"


class Initiator():
    def __init__(self, M, N, v_max, **kwargs):
        self.N = N
        self.M = M
        self.initiators = []
        self.preliminary_tracks = []
        self.v_max = v_max
        self.gamma = tracking_parameters['gamma']
        self.last_timestamp = None
        self.merge_threshold = 5  # meter
        self.log = logging.getLogger(__name__)
        self.log.info("Initiator ready")

    def getPreliminaryTracksString(self):
        return " ".join([str(e) for e in self.preliminary_tracks])

    def processMeasurements(self, measurement_list):
        self.log.debug("processMeasurements " + str(measurement_list.measurements.shape[0]))
        unused_indices, initial_targets = self._processPreliminaryTracks(measurement_list)
        unused_indices = self._processInitiators(unused_indices, measurement_list)
        self._spawnInitiators(unused_indices, measurement_list)
        self.last_timestamp = measurement_list.time
        initial_targets = _merge_similar_targets(initial_targets, self.merge_threshold)
        self.log.debug("new initial targets " + str(len(initial_targets)))
        return initial_targets

    def _processPreliminaryTracks(self, measurement_list):
        newInitialTargets = []
        time = measurement_list.time
        measurement_array = measurement_list.measurements
        self.log.debug("_processPreliminaryTracks " + str(len(self.preliminary_tracks)))

        # Check for something to work on
        n1 = len(self.preliminary_tracks)
        n2 = measurement_array.shape[0]
        n3 = measurement_array.size
        if n1 == 0 or n2 == 0 or n3 == 0:
            return np.arange(n2).tolist(), newInitialTargets

        # Predict position
        dt = time - self.last_timestamp
        F = pv.Phi(dt)
        Q = pv.Q(dt)
        for track in self.preliminary_tracks:
            track.predict(F, Q)
        predicted_states = [track.get_predicted_state_and_clear()
                            for track in self.preliminary_tracks]

        # Calculate delta matrix
        delta_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i, predicted_state in enumerate(predicted_states):
            for j, measurement in enumerate(measurement_array):
                predicted_measurement = pv.H.dot(predicted_state)
                delta_vector = measurement - predicted_measurement
                distance = np.linalg.norm(delta_vector)
                P_bar = self.preliminary_tracks[i].covariance
                S = pv.H.dot(P_bar).dot(pv.H.T) + pv.R_RADAR()
                S_inv = np.linalg.inv(S)
                K = P_bar.dot(pv.H.T).dot(S_inv)
                self.preliminary_tracks[i].K = K
                nis = delta_vector.T.dot(S_inv).dot(delta_vector)
                inside_gate = nis < self.gamma
                delta_matrix[i, j] = distance if inside_gate else np.Inf

        # Assign measurements
        assignments = _solve_global_nearest_neighbour(delta_matrix)

        # Update tracks
        for track_index, meas_index in assignments:
            P_bar = self.preliminary_tracks[track_index].covariance
            K = self.preliminary_tracks[track_index].K
            delta_vector = measurement_array[meas_index] - pv.H.dot(predicted_states[track_index])
            filtered_state = predicted_states[track_index] + K.dot(delta_vector)
            P_hat = P_bar - K.dot(pv.H).dot(P_bar)
            self.preliminary_tracks[track_index].estimates.append(filtered_state)
            self.preliminary_tracks[track_index].covariance = P_hat
            self.preliminary_tracks[track_index].m += 1
            self.preliminary_tracks[track_index].measurement_index = meas_index

        # Add dummy measurement to un-assigned tracks, and increase covariance
        assigned_track_indices = [assignment[0] for assignment in assignments]
        unassigned_track_indices = [track_index
                                    for track_index in range(len(self.preliminary_tracks))
                                    if track_index not in assigned_track_indices]
        for track_index in unassigned_track_indices:
            self.preliminary_tracks[track_index].estimates.append(
                predicted_states[track_index])

        # Increase all N
        for track in self.preliminary_tracks:
            track.n += 1

        self.log.debug(self.getPreliminaryTracksString())
        removeIndices = []
        for track_index, track in enumerate(self.preliminary_tracks):
            track_status = track.mn_analysis(self.M, self.N)
            if track_status == DEAD:
                self.log.debug("Removing DEAD track " + str(track_index))
                removeIndices.append(track_index)
            elif track_status == CONFIRMED:
                self.log.debug("Removing CONFIRMED track " + str(track_index))
                assert len(track.estimates) == self.N + 1
                new_target = Target(time,
                                    None,
                                    np.array(track.estimates[-1]),
                                    track.covariance,
                                    measurementNumber=track.measurement_index+1)
                self.log.debug("Spawning new (initial) Target: " + str(new_target))
                newInitialTargets.append(new_target)
                removeIndices.append(track_index)
        for i in reversed(removeIndices):
            self.preliminary_tracks.pop(i)
        if removeIndices: self.log.debug(self.getPreliminaryTracksString())

        used_indices = [assignment[1] for assignment in assignments]
        unused_indices = [index
                          for index in np.arange(n2)
                          if index not in used_indices]
        return unused_indices, newInitialTargets

    def _processInitiators(self, unused_indices, measurement_list):
        self.log.debug("_processInitiators " + str(len(self.initiators)))
        time = measurement_list.time
        measurementArray = measurement_list.measurements
        n1 = len(self.initiators)
        n2 = len(unused_indices)
        if n1 == 0 or n2 == 0:
            return unused_indices

        delta_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i, initiator in enumerate(self.initiators):
            for j, meas_index in enumerate(unused_indices):
                delta_vector = measurementArray[meas_index] - initiator.value
                delta_matrix[i, j] = np.linalg.norm(delta_vector)

        dt = time - self.initiators[0].timestamp
        gate_distance = self.v_max * dt
        assignments = _solve_global_nearest_neighbour(delta_matrix, gate_distance)
        assigned_local_indices = [assignment[1] for assignment in assignments]
        used_indices = [unused_indices[j] for j in assigned_local_indices]
        unused_indices = [i for i in unused_indices if i not in used_indices]
        unused_indices.sort()
        assert len(unused_indices) == len(set(unused_indices))
        self.__spawn_preliminary_tracks(measurement_list, assignments)
        return unused_indices

    def _spawnInitiators(self, unused_indices, measurement_list):
        self.log.debug("_spawnInitiators " + str(len(unused_indices)))
        time = measurement_list.time
        measurement_array = measurement_list.measurements
        self.initiators = [Measurement(measurement_array[index], time)
                           for index in unused_indices]

    def __spawn_preliminary_tracks(self, measurement_list, assignments):
        time = measurement_list.time
        measurements = measurement_list.measurements
        self.log.debug(
            "__spawn_preliminary_tracks " + str(len(assignments)) + ("\n" + "\n".join(
                ['{0:} -> {1:}'.format(self.initiators[old_index].value, measurements[new_index])
                 for old_index, new_index in assignments]) if assignments else ""))

        for old_index, new_index in assignments:
            delta_vector = measurements[new_index] - self.initiators[old_index].value
            dt = time - self.initiators[old_index].timestamp
            velocity_vector = delta_vector / dt
            x0 = np.hstack((measurements[new_index], velocity_vector))
            track = PreliminaryTrack(x0, pv.P0)
            self.preliminary_tracks.append(track)


if __name__ == "__main__":
    from pymht.utils.classDefinitions import Position
    import pymht.utils.simulator as sim
    import pymht.models.pv as model

    np.set_printoptions(precision=1, suppress=True)

    print("Test 1")
    deltaMatrix = np.array([[5., 2.], [np.Inf, np.Inf]])
    print("test deltaMatrix\n", deltaMatrix)
    assignment = _solve_global_nearest_neighbour(deltaMatrix, debug=True)
    print("test assignment", assignment)

    print("Test 2")
    seed = 1254
    nTargets = 2
    p0 = Position(0, 0)
    radarRange = 1000  # meters
    meanSpeed = 10  # gausian distribution
    P_d = 1.0
    initialTargets = sim.generateInitialTargets(
        seed, nTargets, p0, radarRange, meanSpeed, P_d)
    nScans = 7
    timeStep = 0.7
    simTime = nScans * timeStep
    simList = sim.simulateTargets(seed, initialTargets, simTime, timeStep, model.Phi(
        timeStep), model.Q(timeStep, 0), model.Gamma)

    lambda_phi = 1e-6
    scanList = sim.simulateScans(seed, simList, timeStep, model.C_RADAR, model.R_RADAR(0),
                                 lambda_phi, radarRange, p0, shuffle=True)

    N_checks = 3
    M_required = 2

    v_max = 17
    initiator = Initiator(M_required, N_checks, v_max, debug=True)

    for scanIndex, measurementList in enumerate(scanList):
        print(measurementList)
        initialTargets = initiator.processMeasurements(measurementList)
        if initialTargets:
            print(scanIndex, end="\t")
            print(*initialTargets, sep="\n", end="\n\n")
            # print(*initialTargets, se)
        else:
            print(scanIndex, [], sep="\t")

        print("-" * 50)
