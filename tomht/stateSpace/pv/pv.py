#!/usr/bin/env python3
import numpy as np

C 				= np.array([[1, 0, 0, 0],	#Also known as "H"
						[0, 1, 0, 0]])
H 				= C
Gamma 			= np.diag([1,1],-2)[:,0:2]	#Disturbance matrix (only velocity)
p 				= np.power(1,2)			#Initial systen state variance
P0 				= np.diag([p,p,p,p])		#Initial state covariance
sigmaR_tracker 	= 0.8		#Measurement standard deviation used in kalman filter
sigmaR_true		= 0.5	#Measurement standard deviation used in radar simulator (+- 1.25m)
sigmaQ_tracker	= 0.5		#Target standard deviation used in kalman filter
sigmaQ_true 	= 0.5	#Tardet standard deviation used in kalman filter
#95% conficence = +- 2.5*sigma

def Q(T,sigmaQ = sigmaQ_tracker):
	return np.eye(2) * np.power(sigmaQ,2) * T 	#Transition/system covariance (process noise)

def R(sigmaR = sigmaR_tracker):
	return np.eye(2) * np.power(sigmaR,2)

def Phi(T):
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])