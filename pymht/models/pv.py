import numpy as np

defaultType = np.float32

C_RADAR = np.array([[1.0, 0, 0, 0],  # Also known as "H_radar"
                    [0, 1.0, 0, 0]], dtype=defaultType)
H_radar = C_RADAR

C_AIS = np.eye(4, dtype=defaultType)

H_ais = C_AIS

# Disturbance matrix (only velocity)
Gamma = np.array(np.diag([1.0, 1.0], -2)[:, 0:2], dtype=defaultType)
p = np.power(1.0, 2)  # Initial systen state variance
P0 = np.array(np.diag([p, p, p, p]), dtype=defaultType)  # Initial state covariance
sigmaR_RADAR_tracker = 2.0  # Measurement standard deviation used in kalman filter
sigmaR_RADAR_true = 2.0
sigmaR_AIS_tracker = 0.5
sigmaR_AIS_true = 0.5
sigmaQ_tracker = 1.0  # Target standard deviation used in kalman filterUnused
sigmaQ_true = 1.0  # Target standard deviation used in kalman filterUnused
# 95% confidence = +- 2.5*sigma
GPS_COVARIANCE_PRECISE = np.copy(P0 * 0.5)


def Q(T, sigmaQ=sigmaQ_tracker):
    # Transition/system covariance (process noise)
    return np.array(np.eye(2) * np.power(sigmaQ, 2) * T, dtype=defaultType)


def R_RADAR(sigmaR=sigmaR_RADAR_tracker):
    return np.array(np.eye(2) * np.power(sigmaR, 2), dtype=defaultType)


def R_AIS(sigmaR=sigmaR_AIS_tracker):
    return np.array(np.eye(4) * np.power(sigmaR, 2), dtype=defaultType)


def Phi(T):
    return np.array([[1.0, 0, T, 0],
                     [0, 1.0, 0, T],
                     [0, 0, 1.0, 0],
                     [0, 0, 0, 1.0]],
                    dtype=defaultType)
