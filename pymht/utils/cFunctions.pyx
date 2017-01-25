import numpy as np
cimport numpy as np

ctypedef np.double_t DTYPE_t

cpdef int binomial(int n, int k):
    return 1 if k == 0 else (0 if n == 0 else binomial(n - 1, k) + binomial(n - 1, k - 1))

cpdef double nllrNoMeasurement(double P_d):
    if P_d == 1:
        return -np.log(1e-6)
    return -np.log(1 - P_d)

cpdef double nllr(double P_d, np.ndarray[DTYPE_t, ndim=1] measurementResidual, double lambda_ex, np.ndarray[DTYPE_t, ndim=2] covariance, np.ndarray[DTYPE_t, ndim=2] invCovariance):
    return (0.5 * (measurementResidual.T.dot(invCovariance).dot(measurementResidual))
            + np.log((lambda_ex * np.sqrt(2 * np.pi *np.linalg.det(covariance))) / P_d))