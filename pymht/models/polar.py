import numpy as np
from .constants import *
from .ais import R as R_AIS
from .ais import Phi as A_AIS
from .ais import C as C_AIS

C_RADAR = np.array([[1.0, 0, 0, 0],
                    [0, 1.0, 0, 0]], dtype=defaultType)
H_radar = C_RADAR

p = 2.5**2  # Initial system state variance
P0 = np.array(np.diag([p, p, 0.3*p, 0.3*p]), dtype=defaultType)  # Initial state covariance

sigma_hdg = 3.0
sigma_speed = 0.8

def R_RADAR(sigmaR=sigmaR_RADAR_tracker):
    return np.array(np.eye(2) * np.power(sigmaR, 2), dtype=defaultType)


def Phi(T):
    return np.array([[1.0, 0, T, 0],
                     [0, 1.0, 0, T],
                     [0, 0, 1.0, 0],
                     [0, 0, 0, 1.0]],
                    dtype=defaultType)
