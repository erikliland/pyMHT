"""
A module with operations useful for Kalman filtering.
"""
import numpy as np


class KalmanFilter(object):
    """
    A Kalman filter class, does filtering for systems of the type:
    x_{k+1} = A*x_k + v_k
    y_k = C*z_k + e_k
    v_k ~ N(0,Q)
    e_k ~ N(0,R)
    x_0 - Initial state
    P_0 - Initial state covariance
    """

    def __init__(self, A, C, **kwargs):
        Q = kwargs.get('Q')
        R = kwargs.get('R')
        x_0 = kwargs.get('x_0')
        P_0 = kwargs.get('P_0')
        T = kwargs.get('T')
        try:
            Gamma = kwargs.get('Gamma', np.eye(Q.shape))
        except AttributeError:
            # TODO: less nasty solution
            Gamma = kwargs.get('Gamma', np.eye(Q(1).shape[0]))

        self.A = A          # Transition matrix
        self.C = C          # Observation matrix
        self.Q = Q          # Process noise covariance
        self.R = R          # Measurement noise covariance
        self.Gamma = Gamma  # Process noise "observability"
        self.x_hat = x_0    # Filtered state
        self.P_hat = P_0    # Filtered state covariance
        self.x_bar = None   # Prediced state
        self.P_bar = None   # Predictes state covariance
        self.T = T          # Sampling period (dynamic if None)
        self.z_hat = None   # Predicted measurement
        self.S = None       # Residual covariance
        self.S_inv = None   # Inverse residual covariance
        self.K = None       # Kalman gain
        self.precalculated = False

    def predict(self, **kwargs):
        """
        Calculate next state estimate without actually updating
        the internal variables
        """
        T = kwargs.get('T')
        A = self.A if self.T is not None else self.A(T)
        Q = self.Q if self.T is not None else self.Q(T)
        x_bar = A.dot(self.x_hat)
        P_bar = A.dot(self.P_hat).dot(A.T) + self.Gamma.dot(Q.dot(self.Gamma.T))
        if not kwargs.get('local', False):
            self.x_bar = x_bar
            self.P_bar = P_bar
        return x_bar, P_bar

    def _precalculateMeasurementUpdate(self, T):
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
            self._precalculateMeasurementUpdate(kwargs.get('T', self.T))
        self.precalculated = False

        if 'y_tilde' in kwargs:
            y_tilde = kwargs.get('y_tilde')
        elif 'y' in kwargs:
            z = kwargs.get('y')
            y_tilde = z - self.z_hat
        else:
            raise ValueError
        x_hat = self.x_bar + self.K.dot(y_tilde)
        P_hat = self.P_bar - self.K.dot(self.C).dot(self.P_bar)
        if not kwargs.get('local', False):
            self.x_hat = x_hat
            self.P_hat = P_hat
        return x_hat, P_hat
