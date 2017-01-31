import numpy as np
cimport numpy as np
from libcpp cimport bool

cpdef int binomial(int n, int k):
    return 1 if k == 0 else (0 if n == 0 else binomial(n - 1, k) + binomial(n - 1, k - 1)) 

cpdef double nllrNoMeasurement(double P_d):
    if P_d == 1:
        return -np.log(1e-6)
    return -np.log(1 - P_d)

cpdef double nllr(double P_d, 
                  np.ndarray[np.double_t, ndim=1] measurementResidual, 
                  double lambda_ex, 
                  np.ndarray[np.double_t, ndim=2] covariance, 
                  np.ndarray[np.double_t, ndim=2] invCovariance):
    return (0.5 * (measurementResidual.T.dot(invCovariance).dot(measurementResidual))
            + np.log((lambda_ex * np.sqrt(2 * np.pi *np.linalg.det(covariance))) / P_d))

cpdef np.ndarray[np.double_t, ndim=2] stackNodeMatrix(np.ndarray[np.uint8_t, ndim=1] indecies, 
                                                  np.ndarray[np.double_t, ndim=2] measurementsResidual, 
                                                  np.ndarray[np.double_t, ndim=2] measurements, 
                                                  np.ndarray[np.double_t, ndim=1] cNLLR):
    cdef array = np.zeros((indecies.shape[0],6))
    array[:,0] = indecies
    array[:,1:3] = measurementsResidual
    array[:,3:5] = measurements
    array[:,5] = cNLLR
    return array

cpdef np.ndarray[np.double_t,ndim=2] cNewMeasurement(np.ndarray[np.double_t,ndim=2] measurements,
                                                     np.ndarray[np.double_t,ndim=1] z_hat,
                                                     np.ndarray[np.double_t,ndim=2] S, 
                                                     np.ndarray[np.double_t,ndim=2] S_inv,
                                                     double eta2, 
                                                     double P_d, 
                                                     double lambda_ex):
    cdef np.ndarray[np.double_t,ndim=2] measurementsResidual = measurements - z_hat
    cdef np.ndarray[np.double_t,ndim=1] nis = np.sum(measurementsResidual.dot(S_inv) * measurementsResidual, axis=1)
    cdef np.ndarray[np.int64_t,ndim=1, cast = True] gatedFilter = np.less_equal(nis,np.ones(measurements.shape[0])*eta2)
    cdef np.ndarray[np.int_t,ndim=1] gatedIndecies = np.where(gatedFilter)[0]
    cdef int nMeasurementInsideGate = gatedIndecies.shape[0]
    cdef np.ndarray[np.double_t,ndim=2] gatedMeasurements = measurements[gatedIndecies]
    cdef np.ndarray[np.double_t,ndim=2] gatedMeasurementResidual = np.array(measurementsResidual[gatedIndecies, :], ndmin=2)
    cdef np.ndarray cNLLR = np.zeros(nMeasurementInsideGate,dtype = np.double)
    cdef int i = 0
    for i in range(nMeasurementInsideGate):
        cNLLR[i] = nllr(P_d, gatedMeasurementResidual[i,:], lambda_ex, S, S_inv)
    if nMeasurementInsideGate > 0:
        print(type(gatedIndecies))
        return stackNodeMatrix(gatedIndecies,gatedMeasurementResidual, gatedMeasurements, cNLLR)
    else:
        return np.array([],ndmin=2)