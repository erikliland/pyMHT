import numpy as np
import pymht.models.pv as pv
from munkres import Munkres

tracking_parameters = {
    'process_covariance': 1,
    'measurement_covariance': 1,
    'v_max': 15,
    'gate_probability': 0.99,
    'detection_probability': 0.9,
    'gamma': 5.99,
}
CONFIRMED = 1
PRELIMINARY = 0
DEAD = -1


def _solve_global_neareast_neightbour(delta_matrix, gate_distance, **kwargs):
    DEBUG = kwargs.get('debug', False)
    # Copy and gating
    if DEBUG: print("delta matrix\n", delta_matrix)
    cost_matrix = np.copy(delta_matrix)
    cost_matrix[cost_matrix > gate_distance] = float('inf')
    if DEBUG: print("cost_matrix\n", cost_matrix)

    # Pre-processing
    valid_matrix = cost_matrix < float('inf')
    if np.all(valid_matrix == False):
        return []
    if DEBUG: print("Valid matrix\n", valid_matrix.astype(int))
    bigM = np.power(10., np.ceil(np.log10(np.sum(cost_matrix[valid_matrix]))) + 1.)
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
    assignments = []
    for preliminary_assignment in preliminary_assignments:
        row = preliminary_assignment[0]
        col = preliminary_assignment[1]
        if valid_matrix[row, col]:
            assignments.append(preliminary_assignment)
        if DEBUG: print("assignments", assignments)
    return assignments


class PreliminaryTrack():
    def __init__(self, measurements):
        self.measurements = measurements
        self.n = 0
        self.m = 0

    def __repr__(self):
        return "({0:}|{1:})".format(self.m, self.n)

    def mn_analysis(self, M, N):
        n = self.n
        m = self.m
        if n >= N and m >= M:
            return CONFIRMED
        elif n >= N and m < M:
            return DEAD
        else:
            return PRELIMINARY

    def get_predicted_measurement(self, now_time):
        if len(self.measurements) == 0:
            raise ValueError("No measurements found in PreliminaryTrack")
        elif len(self.measurements) == 1:
            print("Not enough measurements to predict")
            return self.measurements[0]
        delta_pos = self.measurements[-1].value - self.measurements[-2].value
        delta_time = self.measurements[-1].timestamp - self.measurements[-2].timestamp
        velocity = delta_pos / delta_time  # m/s
        predict_delta = now_time - self.measurements[-1].timestamp
        prediction = self.measurements[-1].value + velocity * predict_delta
        return prediction


class Measurement():
    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp
        self.covariance = pv.R()

    def __repr__(self):
        meas_str = "Measurement: (%.2f, %.2f)" % (self.value[0], self.value[1])
        time_str = "Timestamp: %.2f" % (self.timestamp)
        return meas_str + ", " + time_str

    def initiator_distance(self, center_point, test_point, dt):
        v_max = tracking_parameters['v_max']
        delta_vector = test_point - center_point
        movement_vector = dt * v_max
        d_plus = np.maximum(delta_vector - movement_vector, np.zeros(2))
        d_minus = np.maximum(-delta_vector - movement_vector, np.zeros(2))
        d = d_plus + d_minus
        R = self.covariance
        D = np.dot(d.T, np.dot(np.linalg.inv(R + R), d))
        return D

    def inside_gate(self, new_measurement):
        center_point = self.value
        test_point = new_measurement.value
        dt = new_measurement.timestamp - self.timestamp
        D = self.initiator_distance(center_point, test_point, dt)
        start_preliminary = D < tracking_parameters['gamma']
        return start_preliminary


class Initiator():
    def __init__(self, M, N,**kwargs):
        self.N = N
        self.M = M
        self.initiators = []
        self.preliminary_tracks = []
        self.DEBUG = kwargs.get('debug',False)

    def printPreliminaryTracks(self):
        print("preliminary tracks [{}]".format(len(self.preliminary_tracks)),
              *self.preliminary_tracks, sep="\t")

    def processMeasurements(self, measurement_list):
        time = measurement_list.time
        measurements = measurement_list.measurements
        if self.DEBUG: print("processMeasurements", measurements.shape[0])
        unused_indices, initial_targets = self._processPreliminaryTracks(measurement_list)
        unused_indices = self._processInitiators(unused_indices, measurement_list)
        self._processUnusedMeasurements(unused_indices, measurement_list)
        if self.DEBUG: print("initial targets", len(initial_targets))
        if self.DEBUG: print("-" * 50)
        return initial_targets

    def _processPreliminaryTracks(self, measurement_list, gate_distance=30):
        newInitialTargets = []
        time = measurement_list.time
        measurement_array = measurement_list.measurements
        if self.DEBUG: print("_processPreliminaryTracks", len(self.preliminary_tracks))
        # Predict position
        predicted_measurements = [track.get_predicted_measurement(time)
                                  for track in self.preliminary_tracks]
        # Calculate delta matrix
        n1 = len(predicted_measurements)
        n2 = measurement_array.shape[0]
        n3 = measurement_array.size
        if n1 == 0 or n2 == 0 or n3 == 0:
            return np.arange(n2).tolist(), newInitialTargets
        delta_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i, predicted_measurement in enumerate(predicted_measurements):
            for j, measurement in enumerate(measurement_array):
                delta_vector = measurement - predicted_measurement
                delta_matrix[i, j] = np.linalg.norm(delta_vector)
        # Assign measurements
        assignments = _solve_global_neareast_neightbour(delta_matrix, gate_distance)
        # Update tracks
        for track_index, meas_index in assignments:
            self.preliminary_tracks[track_index].measurements.append(Measurement(measurement_array[meas_index], time))
            self.preliminary_tracks[track_index].m += 1
        # Increase all N
        for track in self.preliminary_tracks:
            track.n += 1

        if self.DEBUG: self.printPreliminaryTracks()
        removeIndices = []
        for track_index, track in enumerate(self.preliminary_tracks):
            track_status = track.mn_analysis(self.M, self.N)
            if track_status == DEAD:
                if self.DEBUG: print("Removing DEAD track", track_index)
                removeIndices.append(track_index)
            elif track_status == CONFIRMED:
                if self.DEBUG: print("Removing CONFIRMED track", track_index)
                assert len(track.measurements) == self.N +2
                newInitialTargets.append(track.measurements)
                removeIndices.append(track_index)
        for i in reversed(removeIndices):
            self.preliminary_tracks.pop(i)
        if removeIndices and self.DEBUG: self.printPreliminaryTracks()

        used_indices = [assignment[1] for assignment in assignments]
        unused_indices = [index
                          for index in np.arange(n2)
                          if index not in used_indices]
        return unused_indices, newInitialTargets

    def _processInitiators(self, unused_indices, measurement_list, gate_distance=30):
        if self.DEBUG: print("_processInitiators", len(self.initiators))
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
        assignments = _solve_global_neareast_neightbour(delta_matrix, gate_distance, debug=False)
        assigned_local_indices = [assignment[1] for assignment in assignments]
        used_indices = [unused_indices[j] for j in assigned_local_indices]
        unused_indices = [i for i in unused_indices if i not in used_indices]
        unused_indices.sort()
        assert len(unused_indices) == len(set(unused_indices))
        self.__spawn_preliminary_tracks(measurement_list, assignments)
        return unused_indices

    def _processUnusedMeasurements(self, unused_indices, measurement_list):
        if self.DEBUG: print("_processUnusedMeasurements", len(unused_indices))
        time = measurement_list.time
        measurement_array = measurement_list.measurements
        self.initiators = [Measurement(measurement_array[index], time)
                           for index in unused_indices]

    def __spawn_preliminary_tracks(self, measurement_list, assignments):
        if self.DEBUG: print("__spawn_preliminary_tracks", len(assignments))
        time = measurement_list.time
        measurements = measurement_list.measurements
        for old_index, new_index in assignments:
            m1 = self.initiators[old_index]
            m2 = Measurement(measurements[new_index], time)
            track = PreliminaryTrack([m1, m2])
            self.preliminary_tracks.append(track)


if __name__ == "__main__":
    from pymht.utils.classDefinitions import Position
    import pymht.utils.radarSimulator as sim
    import pymht.models.pv as model

    seed = 1254
    nTargets = 1
    p0 = Position(0, 0)
    radarRange = 1000  # meters
    meanSpeed = 10  # gausian distribution
    P_d = 0.7
    initialTargets = sim.generateInitialTargets(
        seed, nTargets, p0, radarRange, meanSpeed, P_d)
    nScans = 15
    timeStep = 1.0
    simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, model.Phi(
        timeStep), model.Q(timeStep, model.sigmaQ_true), model.Gamma)

    lambda_phi = 0
    scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true),
                                 lambda_phi, radarRange, p0, shuffle=True)

    N_checks = 3
    M_required = 2

    initiator = Initiator(M_required, N_checks)

    for scanIndex, measurementList in enumerate(scanList):
        initialTargets = initiator.processMeasurements(measurementList)
        if initialTargets:
            print(scanIndex, end = "\t")
            # print(*initialTargets, sep = "\n", end = "\n\n")
            print(initialTargets)
        else:
            print(scanIndex, [], sep = "\t")
