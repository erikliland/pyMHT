from pymht.utils import pyKalman
import numpy as np
from pymht.models import pv

dT = 1.0
x_0 = np.zeros(4)
P_0 = pv.P0
A = pv.Phi(dT)
C = pv.C
Gamma = pv.Gamma
Q = pv.Q(dT)
R = pv.R()


def test_KalmanFilter_class():
    kf = pyKalman.KalmanFilter(x_0, P_0, A, C, Gamma, Q, R)
    y = np.ones(2)
    kf.filter(y=y)
    kf2 = kf.filterAndCopy()
    y_tilde = y
    kf3 = kf.filterAndCopy(y_tilde)


def test_numpyPredict():
    n = 10
    x_0_list = np.array([x_0, ] * n)
    P_0_list = np.array([P_0, ] * n)
    x_bar_list, P_bar_list, z_hat_list, S_list, S_inv_list, K_list, P_hat_list = pyKalman.numpyPredict(
        A, C, Q, R, Gamma, x_0_list, P_0_list)


def test_numpyFilter():
    gated_x_hat_list = [kalman.numpyFilter(x_bar_list[i],
                                           K_list[i],
                                           gated_z_tilde_list[i])
                        for i in range(nNodes)]
