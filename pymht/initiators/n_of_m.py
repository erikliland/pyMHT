import numpy as np
from munkres import Munkres

# import matlab.engine
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


def _solve_initial_association(delta_matrix, gate_distance):
    # Copy and gating
    print("delta matrix\n", delta_matrix)
    cost_matrix = np.copy(delta_matrix)
    cost_matrix[cost_matrix > gate_distance] = float('inf')
    print("cost_matrix\n", cost_matrix)

    # Pre-processing
    valid_matrix = cost_matrix < float('inf')
    print("Valid matrix\n", valid_matrix.astype(int))
    bigM = np.power(10., np.ceil(np.log10(np.sum(cost_matrix[valid_matrix]))) + 1.)
    cost_matrix[np.logical_not(valid_matrix)] = bigM
    print("Modified cost matrix\n", cost_matrix)

    validCol = np.any(valid_matrix, axis=0)
    validRow = np.any(valid_matrix, axis=1)
    print("validCol", validCol)
    print("validRow", validRow)
    nRows = int(np.sum(validRow))
    nCols = int(np.sum(validCol))
    n = max(nRows, nCols)
    print("nRows, nCols, n", nRows, nCols, n)

    maxv = 10. * np.max(cost_matrix[valid_matrix])
    print("maxv", maxv)

    rows = np.arange(nRows)
    cols = np.arange(nCols)
    dMat = np.zeros((n, n)) + maxv
    dMat[np.ix_(rows, cols)] = cost_matrix[np.ix_(validRow, validCol)]
    print("dMat\n", dMat)

    # Assignment
    from munkres import Munkres
    m = Munkres()
    preliminary_assignments = m.compute(dMat.tolist())
    print("preliminary assignments", preliminary_assignments)

    # Post-processing
    assignments = []
    for preliminary_assignment in preliminary_assignments:
        row = preliminary_assignment[0]
        col = preliminary_assignment[1]
        if valid_matrix[row,col]:
            assignments.append(preliminary_assignment)
    print("assignments", assignments)
    return  assignments

class PreliminaryTrack():
    def __init__(self, measurement):
        self.measurements = [measurement]
        self.n = 0
        self.m = 0


class Measurement():
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


class Estimate(object):
    def __init__(self, t, mean, covariance, is_posterior=False, track_index=None):
        # If it is a posterior, it should be set accordingly
        H, R = cartesian_measurement()
        self.timestamp = t
        self.measurements = []
        self.H, self.R = cartesian_measurement()
        self.est_prior = mean
        self.cov_prior = covariance
        self.S = np.dot(H, np.dot(covariance, H.T)) + R
        self.z_hat = np.dot(H, self.est_prior)
        if is_posterior:
            self.est_posterior = mean
            self.cov_posterior = covariance
        if track_index is not None:
            self.track_index = track_index
        else:
            self.track_index = -1

    def __repr__(self):
        ID_str = "Track ID: %d" % (self.track_index)
        timestamp_str = "Timestamp: %.2f" % self.timestamp
        return ID_str + ", " + timestamp_str

    def inside_gate(self, measurement):
        z = measurement.value
        nu = z - self.z_hat
        nis = np.dot(nu.T, np.dot(np.linalg.inv(self.S), nu))
        inside_gate = nis.squeeze() < tracking_parameters['gamma']
        return inside_gate

    def store_measurement(self, measurement):
        self.measurements.append(measurement)

    def step_measurement(self):
        if len(self.measurements) > 0:
            self.pdaf_step()
        else:
            self.trivial_step()

    def pdaf_step(self):
        N_measurements = len(self.measurements)
        z_hat = np.dot(self.H, self.est_prior)
        P_G = tracking_parameters['gate_probability']
        P_D = tracking_parameters['detection_probability']
        gamma = tracking_parameters['gamma']
        b = 2 / gamma * N_measurements * (1 - P_D * P_G) / P_D
        e = np.zeros(N_measurements)
        innovations = np.zeros((2, N_measurements))
        for i in range(N_measurements):
            innovations[:, i] = self.measurements[i].value - z_hat
            e[i] = np.exp(np.dot(innovations[:, i], np.dot(np.linalg.inv(self.S), innovations[:, i])))
        betas = np.hstack((e, b))
        betas = betas / (1. * np.sum(betas))
        gain = np.dot(self.cov_prior, np.dot(self.H.T, np.linalg.inv(self.S)))
        total_innovation = np.zeros(2)
        cov_terms = np.zeros((2, 2))
        for i in range(N_measurements):
            innov = innovations[:, i]
            total_innovation += betas[i] * innov
            innov_vec = innov.reshape((2, 1))
            cov_terms += betas[i] * np.dot(innov_vec, innov_vec.T)
        self.est_posterior = self.est_prior + np.dot(gain, total_innovation)
        total_innovation_vec = total_innovation.reshape((2, 1))
        cov_terms = cov_terms - np.dot(total_innovation_vec, total_innovation_vec.T)
        soi = np.dot(gain, np.dot(cov_terms, gain.T))
        P_c = self.cov_prior - np.dot(gain, np.dot(self.S, gain.T))
        self.cov_posterior = betas[-1] * self.cov_prior + (1 - betas[-1]) * P_c + soi
        self.cov_posterior = 0.5 * (self.cov_posterior + self.cov_posterior.T)

    def trivial_step(self):
        self.est_posterior = self.est_prior
        self.cov_posterior = self.cov_prior

    def to_message(self, header):
        def measurement2msg(meas):
            val = automsg.Vector2(x=meas.value[0], y=meas.value[1])
            cov = automsg.Covariance2(var_x=meas.covariance[0, 0], var_y=meas.covariance[1, 1],
                                      cor_xy=meas.covariance[1, 0])
            return automsg.RadarMeasurement(
                header=header,
                value=val,
                covariance=cov,
                coordinate_system=automsg.RadarMeasurement.CARTESIAN)

        def est2msg(est, cov):
            # Transform x=[N v_N E v_E] to pos and vel
            T_pos = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
            T_vel = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
            est_pos = T_pos.dot(est)
            est_vel = T_vel.dot(est)
            cov_pos = T_pos.dot(cov.dot(T_pos.T))
            cov_vel = T_vel.dot(cov.dot(T_vel.T))
            cov_cross = T_pos.dot(cov.dot(T_vel.T))
            V2 = lambda x: automsg.Vector2(x=x[0], y=x[1])
            C2 = lambda c: automsg.Covariance2(var_x=c[0, 0], var_y=c[1, 1], cor_xy=c[0, 1])
            return automsg.KinematicEstimate(
                pos_est=V2(est_pos),
                vel_est=V2(est_vel),
                pos_cov=C2(cov_pos),
                vel_cov=C2(cov_vel),
                pos_vel_corr=C2(cov_cross))

        measurements_msg = [measurement2msg(measurement) for measurement in self.measurements]
        prior_msg = est2msg(self.est_prior, self.cov_prior)
        posterior_msg = est2msg(self.est_posterior, self.cov_posterior)
        return automsg.RadarEstimate(
            header=header,
            measurements=measurements_msg,
            prior=prior_msg,
            posterior=posterior_msg,
            track_id=self.track_index)

    def get_gate_ellipse_scale(self):
        S_inv = np.linalg.inv(self.S)
        gamma = tracking_parameters['gamma']
        return np.sqrt(gamma / S_inv[0, 0]), np.sqrt(gamma / S_inv[1, 1])

    @classmethod
    def from_estimate(cls, t, old_estimate):
        dt = t - old_estimate.timestamp
        F, Q = DWNA_model(dt)
        mean = np.dot(F, old_estimate.est_posterior)
        cov = np.dot(F, np.dot(old_estimate.cov_posterior, F.T)) + Q
        return cls(t, mean, cov, track_index=old_estimate.track_index)

    @classmethod
    def from_measurement(cls, old_measurement, new_measurement):
        H, R = cartesian_measurement()
        t1 = old_measurement.timestamp
        t2 = new_measurement.timestamp
        dt = t2 - t1
        F, _ = DWNA_model(dt)
        H_s = np.vstack((H, np.dot(H, F)))
        z_s = np.hstack((old_measurement.value, new_measurement.value))
        R_s = block_diag(R, R)
        S_s = np.dot(H_s.T, np.dot(np.linalg.inv(R_s), H_s))
        S_s_inv = np.linalg.inv(S_s)
        est_x1 = np.dot(np.dot(S_s_inv, np.dot(H_s.T, np.linalg.inv(R_s))), z_s)
        est_x2 = np.dot(F, est_x1)
        cov_x1 = S_s_inv
        cov_x2 = np.dot(F, np.dot(S_s_inv, F.T))
        est_1 = cls(t1, est_x1, cov_x1, is_posterior=True)
        est_2 = cls(t2, est_x2, cov_x2, is_posterior=True)
        est_1.store_measurement(old_measurement)
        est_2.store_measurement(new_measurement)
        return est_1, est_2

    @classmethod
    def from_message(cls, msg):
        t = msg.header.stamp.to_sec()
        track_id = msg.track_id

        def kinmsg2posVel(msg):
            pos_est = msg.pos_est
            vel_est = msg.vel_est
            est = np.array([pos_est.x, pos_est.y, vel_est.x, vel_est.y])

            def msg2cov(msg):
                return np.array([[msg.var_x, msg.cor_xy], [msg.cor_xy, msg.var_y]])

            pos_cov = msg2cov(msg.pos_cov)
            vel_cov = msg2cov(msg.vel_cov)
            crs_cov = msg2cov(msg.pos_vel_corr)
            cov = block_diag(pos_cov, vel_cov)
            cov[:2, 2:] = crs_cov
            cov[2:, :2] = crs_cov.T
            return est, cov

        def meamsg2meas(msg):
            timestamp = msg.header.stamp.to_sec()
            value = np.array([msg.value.x, msg.value.y])
            return Measurement(value, timestamp)

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        est_prior, cov_prior = kinmsg2posVel(msg.prior)
        est_posterior, cov_posterior = kinmsg2posVel(msg.posterior)
        est_prior = H.dot(est_prior)
        cov_prior = H.dot(cov_prior.dot(H.T))
        est_posterior = H.dot(est_posterior)
        cov_posterior = H.dot(cov_posterior.dot(H.T))
        est = cls(t, est_prior, cov_prior, track_index=track_id)
        est.est_posterior = est_posterior
        est.cov_posterior = cov_posterior
        est.measurements = [meamsg2meas(m) for m in msg.measurements]
        return est

class Initiator():
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.initiators = []
        self.preliminary_tracks = []

    def processMeasurements(self, measurementList):
        time = measurementList.time
        measurements = measurementList.measurements
        unused_measurements, new_initial_targets = self._processPreliminaryTracks(
            measurements)
        unused_measurements = self._processInitiators(unused_measurements, time)
        self.initiators = [Measurement(meas, time) for meas in unused_measurements]
        print("Adding", len(self.initiators), "new measurements to initial tracks")
        print("-" * 50)
        return new_initial_targets

    def _processInitiators(self, measurements, time):
        print(" _processInitiators")
        print("Initial tracks\n", self.initiators)
        print("Measurements", type(measurements), "\n", measurements)
        n1 = len(self.initiators)
        if n1 == 0:
            used_measurements_indices = np.zeros_like(measurements, dtype=np.int)
            unused_measurements = np.ma.array(
                measurements, mask=used_measurements_indices)
            return unused_measurements
        n2 = measurements.shape[0]

        delta_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i, initiator in enumerate(self.initiators):
            for j, measurement in enumerate(measurements):
                delta_vector = measurement - initiator
                delta_matrix[i, j] = np.linalg.norm(delta_vector)
        gate_distance = 30
        assignments = _solve_initial_association(delta_matrix, gate_distance)
        used_measurements_indices = [assignment[1] for assignment in assignments]
        print("Used measurement indecies", used_measurements_indices)
        self._spaw_preliminary_tracks(measurements, assignments)
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indices)
        return unused_measurements

    def _processPreliminaryTracks(self, measurements):
        print("_processPreliminaryTracks")
        for preliminaryTrack in self.preliminary_tracks:
            print("This function is not implemented yet!")
        used_measurements_indecies = np.zeros_like(measurements, dtype=np.bool)
        newInitialTargets = []
        unused_measurements = np.ma.array(measurements, mask=used_measurements_indecies)
        return unused_measurements, newInitialTargets

    def _spaw_preliminary_tracks(self, measurements, assignments):
        print("_spaw_preliminary_tracks")
        for old_index, new_index in assignments:
            e1, e2 = Estimate.from_measurement(self.initiators[old_index],measurements[new_index])
            track = PreliminaryTrack.from_estimates(e1,e2)
            self.preliminary_tracks.append(track)

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
    nScans = 2
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
