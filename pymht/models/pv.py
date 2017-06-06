import numpy as np
from .constants import *
from .ais import R as R_AIS
from .ais import Phi as A_AIS
from .ais import C as C_AIS

C_RADAR = np.array([[1.0, 0, 0, 0],  # Also known as "H_radar"
                    [0, 1.0, 0, 0]], dtype=defaultType)
H_radar = C_RADAR

# Disturbance matrix (only velocity)
p = 2.5**2  # Initial system state variance
P0 = np.array(np.diag([p, p, 0.3*p, 0.3*p]), dtype=defaultType)  # Initial state covariance

GPS_COVARIANCE_PRECISE = np.copy(P0 * 0.5)

def Q(T, sigmaQ=sigmaQ_tracker):
    # Transition/system covariance (process noise)
    # return np.array(np.eye(2) * np.power(sigmaQ, 2) * T, dtype=defaultType)
    return np.array([[T**4./4., 0., T**3./3., 0.],
                     [0., T**4./4., 0., T**3./3.],
                     [T**3./3., 0., T**2., 0.],
                     [0., T**3./3., 0., T**2.]], dtype=defaultType)* sigmaQ


def R_RADAR(sigmaR=sigmaR_RADAR_tracker):
    return np.array(np.eye(2) * np.power(sigmaR, 2), dtype=defaultType)

def Phi(T):
    return np.array([[1.0, 0, T, 0],
                     [0, 1.0, 0, T],
                     [0, 0, 1.0, 0],
                     [0, 0, 0, 1.0]],
                    dtype=defaultType)

if __name__== '__main__':
    print("A(1)\n", Phi(1))
    print("C_Radar\n", C_RADAR)
    print("P0\n", P0)
    print("Q(2)\n", Q(2))
    print("R_Radar\n", R_RADAR())
    print("A P0 A^T + Q(1)\n", Phi(1).dot(P0).dot(Phi(1).T))