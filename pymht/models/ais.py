import numpy as np
from .constants import *

C = np.eye(N=nObsDim_AIS, M=nDimState)

sigmaR_AIS_true_highAccuracy = 1.0
sigmaR_AIS_true_lowAccuracy = 3.0

def R(highAccuracy):
    if highAccuracy:
        return np.array(np.eye(nObsDim_AIS) * np.power(sigmaR_AIS_true_highAccuracy, 2), dtype=defaultType)
    else:
        return np.array(np.eye(nObsDim_AIS) * np.power(sigmaR_AIS_true_lowAccuracy, 2), dtype=defaultType)

def Phi(T):
    return np.array([[1.0, 0, T, 0],
                     [0, 1.0, 0, T],
                     [0, 0, 1.0, 0],
                     [0, 0, 0, 1.0]],
                    dtype=defaultType)