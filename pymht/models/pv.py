import numpy as np

defaultType = np.float32
nObsDim_AIS = 4
nDimState = 4

C_RADAR = np.array([[1.0, 0, 0, 0],  # Also known as "H_radar"
                    [0, 1.0, 0, 0]], dtype=defaultType)
H_radar = C_RADAR

C_AIS = np.eye(N=nObsDim_AIS, M=nDimState)

H_ais = C_AIS


# Disturbance matrix (only velocity)
# Gamma = np.array(np.diag([1.0, 1.0], -2)[:, 0:2], dtype=defaultType)
p = 2.5**2  # Initial system state variance
P0 = np.array(np.diag([p, p, 0.3*p, 0.3*p]), dtype=defaultType)  # Initial state covariance
sigmaR_RADAR_tracker = 2.5  # Measurement standard deviation used in kalman filter
sigmaR_RADAR_true = 2.5
sigmaR_AIS_tracker = 1.0
sigmaR_AIS_true = 1.0
sigmaQ_tracker = 1.0  # Target standard deviation used in kalman filterUnused
sigmaQ_true = 1.0  # Target standard deviation used in kalman filterUnused
# 95% confidence = +- 2.5*sigma
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


def R_AIS(sigmaR=sigmaR_AIS_tracker):
    return np.array(np.eye(nObsDim_AIS) * np.power(sigmaR, 2), dtype=defaultType)


def Phi(T):
    return np.array([[1.0, 0, T, 0],
                     [0, 1.0, 0, T],
                     [0, 0, 1.0, 0],
                     [0, 0, 0, 1.0]],
                    dtype=defaultType)

if __name__== '__main__':
    print("A(1)\n", Phi(1))
    print("C_Radar\n", C_RADAR)
    print("C_AIS\n", C_AIS)
    # print("Gamma\n", Gamma)
    print("P0\n", P0)
    print("Q(2)\n", Q(2))
    print("R_Radar\n", R_RADAR())
    print("R_AIS\n", R_AIS())
    # print("Gamma Q(1) GammaT\n",Gamma.dot(Q(1)).dot(Gamma.T))
    print("A P0 A^T + Q(1)\n", Phi(1).dot(P0).dot(Phi(1).T))