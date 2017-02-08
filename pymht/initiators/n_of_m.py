import numpy as np

tracking_parameters = {
    'process_covariance': 1,
    'measurement_covariance': 1,
    'v_max': 1,
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
    return H, R()*2

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
        d_plus = np.maximum(test_point - center_point - dt * v_max, np.zeros(2))
        d_minus = np.maximum(-(test_point - center_point + dt * v_max), np.zeros(2))
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
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.initial_tracks = []
        self.preliminary_tracks = []

    def processMeasurements(self, measurementList):
        time = measurementList.time
        measurements = measurementList.measurements
        unused_measurements, new_initial_targets = self._processPreliminaryTracks(measurements)
        unused_measurements = self._processInitialTracks(unused_measurements, time)
        self.initial_tracks = np.copy(unused_measurements)
        return new_initial_targets

    def _processInitialTracks(self, measurements, time):
        deltaTensor = self.

        used_measurements_indices = np.zeros_like(measurements, dtype=np.int)
        for measurementIndex, measurement in enumerate(measurements):
            print("Working on measurement index", measurementIndex)
            for trackIndex, initialTrack in enumerate(self.initial_tracks):
                print("Checking against track index:", trackIndex)
                if initialTrack.inside_gate(Measurement(measurement, time)):
                    self.preliminary_tracks.append(PreliminaryTrack(measurement))
                    used_measurements_indices[measurementIndex] = np.ones(2, dtype=np.bool)
                    print("Adding measIndex", measurementIndex)
                    break
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indices)
        return unused_measurements

    def _processPreliminaryTracks(self, measurements):
        for preliminaryTrack in self.preliminary_tracks:
            print("This function is not implemented yet!")
        used_measurements_indecies = np.zeros_like(measurements, dtype=np.bool)
        newInitialTargets = []
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indecies)
        return unused_measurements, newInitialTargets
