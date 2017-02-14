import numpy as np
from munkres import munkres
import matlab.engine
np.set_printoptions(precision=0, suppress=True)

tracking_parameters = {
    'process_covariance': 1,
    'measurement_covariance': 1,
    'v_max': 15,
    'N_checks': 3,
    'M_required': 2,
    'gate_probability': 0.99,
    'detection_probability': 0.9,
    'gamma': 5.99,
}


def cartesian_measurement():
    from pymht.models.pv import H, R
    # H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
    # R = tracking_parameters['measurement_covariance']*np.identity(2)
    return H, R() * 2


class PreliminaryTrack():

    def __init__(self, measurement):
        self.measurements = [measurement]
        self.n = 0
        self.m = 0


class Measurement(object):

    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp
        _, self.covariance = cartesian_measurement()

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


def _solve_initial_association(delta_matrix, gate_distance, eng):
    print("delta matrix\n", delta_matrix)
    delta_matrix_copy = np.copy(delta_matrix)
    delta_matrix_copy[delta_matrix_copy > gate_distance] = float('inf')
    print("delta_matrix_copy\n", delta_matrix_copy)
    result = eng.munkres(matlab.double(delta_matrix_copy.tolist()), nargout=2)
    print("result\n", result)
    matlab_indecies = result[0]
    numpy_indecies = np.array(matlab_indecies, dtype=np.int)
    numpy_indecies -= 1
    return numpy_indecies[0]


class Initiator():

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.initial_tracks = np.empty((0, 2), dtype=np.float32)
        self.preliminary_tracks = []
        self.eng = matlab.engine.start_matlab()

    def processMeasurements(self, measurementList):
        time = measurementList.time
        measurements = measurementList.measurements
        unused_measurements, new_initial_targets = self._processPreliminaryTracks(
            measurements)
        unused_measurements = self._processInitialTracks(unused_measurements, time)
        self.initial_tracks = np.copy(unused_measurements)
        print("Adding", len(self.initial_tracks), "new measurements to initial tracks")
        print("-" * 50)
        return new_initial_targets

    def _processInitialTracks(self, measurements, time):
        print("Initial tracks", type(self.initial_tracks), "\n", self.initial_tracks)
        print("Measurements", type(measurements), "\n", measurements)
        n1 = self.initial_tracks.shape[0]
        if n1 == 0:
            used_measurements_indices = np.zeros_like(measurements, dtype=np.int)
            unused_measurements = np.ma.array(
                measurements, mask=used_measurements_indices)
            return unused_measurements
        n2 = measurements.shape[0]

        delta_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i, initiator in enumerate(self.initial_tracks):
            for j, measurement in enumerate(measurements):
                delta_vector = measurement - initiator
                delta_matrix[i, j] = np.linalg.norm(delta_vector)
        gate_distance = 30
        used_measurements_indices = _solve_initial_association(
            delta_matrix, gate_distance, self.eng)
        print("Used measurement indecies\n", used_measurements_indices)
        used_measurement_pairs = [(initiator_index, measurement_index)
                                  for initiator_index, measurement_index in enumerate(used_measurements_indices)
                                  if measurement_index >= 0]

        self._spaw_preliminary_tracks(used_measurement_pairs)
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indices)
        return unused_measurements

    def _processPreliminaryTracks(self, measurements):
        for preliminaryTrack in self.preliminary_tracks:
            print("This function is not implemented yet!")
        used_measurements_indecies = np.zeros_like(measurements, dtype=np.bool)
        newInitialTargets = []
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indecies)
        return unused_measurements, newInitialTargets

    def _spaw_preliminary_tracks(self, measurements, used_measurements_indecies):
        pass

if __name__ == "__main__":
    from pymht.utils.classDefinitions import MeasurementList, Position
    import pymht.utils.radarSimulator as sim
    import pymht.models.pv as model
    seed = 1254
    nTargets = 2
    p0 = Position(0, 0)
    radarRange = 1000  # meters
    meanSpeed = 10  # gausian distribution
    P_d = 1.0
    initialTargets = sim.generateInitialTargets(
        seed, nTargets, p0, radarRange, meanSpeed, P_d)
    nScans = 3
    timeStep = 1.0
    simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, model.Phi(
        timeStep), model.Q(timeStep, model.sigmaQ_true), model.Gamma)

    lambda_phi = 0
    scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true),
                                 lambda_phi, radarRange, p0, shuffle=False)

    N = 2
    M = 3
    initiator = Initiator(N, M)

    for scanIndex, measurementList in enumerate(scanList):
        initiator.processMeasurements(measurementList)
