import numpy as np

def Phi(T):
	return np.array([[1, 0, T, 0],
					[0, 1, 0, T],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])
C 		= np.array([[1, 0, 0, 0],	#Also known as "H"
					[0, 1, 0, 0]])
H 		= C
Gamma 	= np.diag([1,1],-2)[:,0:2]	#Disturbance matrix (only velocity)
p 		= np.power(1e-1,1)			#Initial systen state variance
P0 		= np.diag([p,p,p,p])		#Initial state covariance
r		= np.power(1e-1,1)			#Measurement variance
rSim 	= np.power(5e-2,1)			#Measurement variance used in radar simulator
q 		= np.power(8e-2,1)			#Velocity variance variance
R 		= np.eye(2) * r 			#Measurement/observation covariance
Rsim 	= np.eye(2) * rSim
def Q(T,qVal = q):
	return np.eye(2) * qVal * T 		#Transition/system covariance (process noise)

