import numpy as np
defaultType = np.float32

nDimState = 4
nObsDim_AIS = 4

sigmaR_RADAR_tracker = 2.5  # Measurement standard deviation used in kalman filter
sigmaR_RADAR_true = 2.5
sigmaQ_tracker = 1.0  # Target standard deviation used in kalman filterUnused
sigmaQ_true = 1.0  # Target standard deviation used in kalman filterUnused
# 95% confidence = +- 2.5*sigma