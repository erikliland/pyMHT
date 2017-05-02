"""
A module with operations useful for Kalman filtering.
"""
import numpy as np


def nllr(lambda_ex, P_d, S_list, nis):
    # assert S_list.shape[0] ==  nis.size, str(S_list.shape) + str(nis.size) + str(nis.shape)
    if lambda_ex == 0:
        print("RuntimeError('lambda_ex' can not be zero.)")
        lambda_ex += 1e-20
    result = (0.5 * nis + np.log((lambda_ex * np.sqrt(np.linalg.det(2 * np.pi * S_list))) / P_d))
    assert result.size == nis.size
    return result


def normalizedInnovationSquared(z_tilde_list, S_inv_list):
    return np.sum(np.matmul(z_tilde_list, S_inv_list) *
                  z_tilde_list,
                  axis=2)

def nis_single(z_tilde, S):
    nis =  z_tilde.dot(np.linalg.inv(S).dot(z_tilde.T))
    return nis


def z_tilde(z_list, z_hat_list, nNodes=1, measDim=2):
    z_tensor = np.array([z_list, ] * nNodes)
    z_hat_tensor = z_hat_list.reshape(nNodes, 1, measDim)
    z_tilde_list = z_tensor - z_hat_tensor
    return z_tilde_list


def numpyFilter(x_bar, K, z_tilde):
    x_bar = x_bar.reshape(1, x_bar.shape[0])
    assert z_tilde.ndim == 2
    assert z_tilde.shape[1] == K.shape[1], str(z_tilde.shape) + str(x_bar.shape)
    assert z_tilde.ndim == 2
    assert K.shape[0] == x_bar.shape[1]
    # assert x_bar.shape == (1, 4), str(x_bar.shape)
    x_hat = x_bar + np.matmul(K, z_tilde.T).T
    assert x_hat.shape[1] == x_bar.shape[1], str(x_hat.shape) + str(x_bar.shape)
    return x_hat


def predict(A, Q, x_0_list, P_0_list):
    assert A.ndim == 2
    assert Q.ndim == 2
    assert x_0_list.ndim == 2
    assert P_0_list.ndim == 3
    x_bar_list = A.dot(x_0_list.T).T
    P_bar_list = (np.matmul(np.matmul(A, P_0_list), A.T) + Q)
    assert x_bar_list.shape == x_0_list.shape, "x_bar ERROR"
    assert P_bar_list.shape == P_0_list.shape, "P_bar ERROR"
    return x_bar_list, P_bar_list


def predict_single(A, Q, x_hat, P_hat):
    x_bar = A.dot(x_hat)
    P_bar = A.dot(P_hat).dot(A.T) + Q
    return x_bar, P_bar

def filter_single(z, x_bar, P_bar, H, R):
    y_tilde = z - H.dot(x_bar)
    S = H.dot(P_bar).dot(H.T)+R
    K = P_bar.dot(H.T).dot(np.linalg.inv(S))
    x_hat = x_bar + K.dot(y_tilde)
    P_hat = P_bar - K.dot(H).dot(P_bar)
    return x_hat, P_hat, S, y_tilde

def precalc(A, C, Q, R, x_bar_list, P_bar_list):
    assert A.ndim == 2
    assert C.ndim == 2
    assert Q.ndim == 2
    assert R.ndim == 2

    nMeasurement, nStates = x_bar_list.shape
    nObservableState = C.shape[0]

    z_hat_list = C.dot(x_bar_list.T).T
    S_list = np.matmul(np.matmul(C, P_bar_list), C.T) + R
    S_inv_list = np.linalg.inv(S_list)
    K_list = np.matmul(np.matmul(P_bar_list, C.T), S_inv_list)
    P_hat_list = P_bar_list - np.matmul(K_list.dot(C), P_bar_list)

    assert z_hat_list.shape == (nMeasurement, nObservableState), "z_hat ERROR"
    assert S_list.shape == (nMeasurement, nObservableState, nObservableState), "S ERROR"
    assert S_inv_list.shape == S_list.shape, "S_inv ERROR"
    assert K_list.shape == (nMeasurement, nStates, nObservableState)
    assert P_hat_list.shape == P_bar_list.shape, "P_hat ERROR"

    return z_hat_list, S_list, S_inv_list, K_list, P_hat_list


class KalmanFilter():
    """
    A Kalman filterUnused class, does filtering for systems of the type:
    x_{k+1} = A*x_k + v_k
    y_k = C_RADAR*z_k + e_k
    v_k ~ N(0,Q)
    e_k ~ N(0,R_RADAR)
    x_0 - Initial state
    P_0 - Initial state covariance
    """

    def __init__(self, x_0, P_0, A, C, Gamma, Q, R, **kwargs):
        # Q = kwargs.get('Q')
        # R_RADAR = kwargs.get('R_RADAR')
        # x_0 = kwargs.get('x_0')
        # P_0 = kwargs.get('P_0')
        # dT = kwargs.get('T')
        # Gamma = kwargs.get(
        #     'Gamma',
        #     np.eye(Q(1).shape[0]) if callable(Q) else np.eye(Q.shape[0]))

        self.A = A  # Transition matrix
        self.C = C  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.Gamma = Gamma  # Process noise "observability"
        self.x_hat = np.copy(x_0)  # Filtered state
        self.P_hat = np.copy(P_0)  # Filtered state covariance
        self.x_bar = None  # Prediced state
        self.P_bar = None  # Predictes state covariance
        # self.dT = np.copy(dT)   # Sampling period (dynamic if None)
        self.z_hat = None  # Predicted measurement
        self.S = None  # Residual covariance
        self.S_inv = None  # Inverse residual covariance
        self.K = None  # Kalman gain
        self.predicted = False
        self.precalculated = False

    def predict(self, **kwargs):
        """
        Calculate next state estimate without actually updating
        the internal variables
        """
        # dT = kwargs.get('T', self.dT)
        A = self.A  # self.A(dT) if callable(self.A) else self.A
        Q = self.Q  # self.Q(dT) if callable(self.Q) else self.Q
        x_bar = A.dot(self.x_hat)
        P_bar = A.dot(self.P_hat).dot(A.T) + self.Gamma.dot(Q.dot(self.Gamma.T))
        if not kwargs.get('local', False):
            self.x_bar = x_bar
            self.P_bar = P_bar
            self.predicted = True
        return x_bar, P_bar

    def _precalculateMeasurementUpdate(self):
        if not self.predicted:
            self.predict()
        self.z_hat = self.C.dot(self.x_bar)
        self.S = self.C.dot(self.P_bar).dot(self.C.T) + self.R
        self.S_inv = np.linalg.inv(self.S)
        self.K = self.P_bar.dot(self.C.T).dot(self.S_inv)
        self.precalculated = True

    def filter(self, **kwargs):
        """
        Filter state with measurement without updating the internal variables
        """
        if not self.precalculated:
            self._precalculateMeasurementUpdate()
        self.precalculated = False

        if 'y_tilde' in kwargs:
            y_tilde = kwargs.get('y_tilde')
        elif 'y' in kwargs:
            z = kwargs.get('y')
            y_tilde = z - self.z_hat
        else:
            x_hat = self.x_bar
            P_hat = self.P_bar
            if not kwargs.get('local', False):
                self.x_hat = x_hat
                self.P_hat = P_hat
            return x_hat, P_hat

        x_hat = self.x_bar + self.K.dot(y_tilde)
        P_hat = self.P_bar - self.K.dot(self.C).dot(self.P_bar)
        if not kwargs.get('local', False):
            self.x_hat = x_hat
            self.P_hat = P_hat
        return x_hat, P_hat

    def filterAndCopy(self, *args):
        if len(args) == 0:
            x_hat = self.x_bar
            P_hat = self.P_bar
        elif len(args) == 1:
            y_tilde = args[0]
            x_hat = self.x_bar + self.K.dot(y_tilde)
            P_hat = self.P_bar - self.K.dot(self.C).dot(self.P_bar)
        else:
            raise ValueError("Invalid number of arguments")
        return KalmanFilter(x_hat, P_hat, self.A, self.C, self.Gamma, self.Q, self.R)
