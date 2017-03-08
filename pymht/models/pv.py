import numpy as np
defaultType = np.float32

C_RADAR = np.array([[1.0, 0, 0, 0],  # Also known as "H"
              [0, 1.0, 0, 0]], dtype=defaultType)
H = C_RADAR


C_AIS = np.eye(4, dtype=defaultType)

# Disturbance matrix (only velocity)
Gamma = np.array(np.diag([1.0, 1.0], -2)[:, 0:2], dtype=defaultType)
p = np.power(1.0, 2)  # Initial systen state variance
P0 = np.array(np.diag([p, p, p, p]), dtype=defaultType)  # Initial state covariance
sigmaR_tracker = 1.0  # Measurement standard deviation used in kalman filter
# Measurement standard deviation used in radar simulator (+- 1.25m)
sigmaR_true = 1.0
sigmaQ_tracker = 1.0  # Target standard deviation used in kalman filter
sigmaQ_true = 0.5  # Tardet standard deviation used in kalman filter
# 95% conficence = +- 2.5*sigma
GPS_COVARIANCE_PRECISE = np.copy(P0*0.5)

def Q(T, sigmaQ=sigmaQ_tracker):
    # Transition/system covariance (process noise)
    return np.array(np.eye(2) * np.power(sigmaQ, 2) * T, dtype=defaultType)


def R(sigmaR=sigmaR_tracker):
    return np.array(np.eye(2) * np.power(sigmaR, 2), dtype=defaultType)


def Phi(T):
    return np.array([[1.0, 0,    T,  0],
                     [0,   1.0,  0,  T],
                     [0,   0,  1.0,  0],
                     [0,   0,    0,  1.0]],
                    dtype=defaultType)
