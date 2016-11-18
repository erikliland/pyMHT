import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.ElementTree as ET



file = "compareResult.xml"
simulations = ET.parse(file).getroot()

fileSet 	= set()
solverSet 	= set()
pdSet 		= set()
nSet 		= set()
lambdaSet	= set()

for simulation in simulations.findall('simulation'):
	att = simulation.attrib
	fileSet.add(att.get("file"))
	solverSet.add(att.get("solver"))
	pdSet.add(att.get("P_d"))
	nSet.add(att.get("N"))
	lambdaSet.add(att.get("lambda_phi"))

for file in fileSet:
	for solver in solverSet:
		for P_d in pdSet:
			for N in nSet:
				lambdaPhiList = []
				for lambdaPhi in lambdaSet:
					print(lambdaPhi)
					print([simulation for simulation in simulations.iter(lambda_phi == lambdaPhi)])
	# 				value 		= lambdaPhi.attrib
	# 				simulation 	= 
	# 				print(simulation)
	# 				nTracks 	= simulation.get("nTracks")
	# 				nLostTracks = lambdaPhi.get("nLostTracks")
	# 				if (nTracks is not None) and (nLostTracks is not None):
	# 					lambdaPhiList.append([value, 1-(nLostTracks/nTracks)])
	# 			if lambdaPhiList:
	# 				ax = figure.add_subplot(111,projection='3d')
	# 				array = np.array(lambdaPhiList, ndmin = 2)
	# 				print(lambdaPhiList)
	# 				# ax.plot(array[:,0], array[:,1],z)
				break
			break
		break
	break

	# figure.show()