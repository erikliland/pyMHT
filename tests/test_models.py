from pymht.models import pv


def test_Q():
    Q_1 = pv.Q(1)
    Q_2 = pv.Q(1, 2)
    assert Q_1.shape == Q_2.shape


def test_R():
    R_1 = pv.R_RADAR()
    R_2 = pv.R_RADAR(2)
    assert R_1.shape == R_2.shape


def test_Phi():
    Phi_1 = pv.Phi(1)
    Phi_2 = pv.Phi(2.0)
    assert Phi_1.shape == Phi_2.shape
