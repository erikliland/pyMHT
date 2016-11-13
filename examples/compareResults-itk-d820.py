import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import matplotlib.pyplot as plt
import tomht.radarSimulator as sim
import numpy as np
import xml.etree.ElementTree as ET

def openRes(filename):
	try:
		f = open(filename)
		trackList = []
		for track in f:
			posStringArray = "".join(track.split()).strip(" ( ) ").split("),(")
			trackArray = []
			for pos in posStringArray:
				trackArray.append( np.array( pos.split(","), dtype = float) )
			trackList.append(np.asarray(trackArray))
		return np.asarray(trackList)
	except:
		return

def openGroundTruth(filename):
	(initialTargets, simList) = sim.importFromFile(filename)
	nTargets = len(initialTargets)
	nTimestep= len(simList)
	targetTracks = np.zeros((nTargets,nTimestep+1,2))
	for target in range(nTargets):
		for timestep in range(nTimestep+1):
			if timestep == 0:
				targetTracks[target,timestep,:] = initialTargets[target].state[0:2]
			else:
				targetTracks[target,timestep,:] = simList[timestep-1][target].state[0:2]
	return np.asarray(targetTracks)

loadLocation = os.path.join("..","data")
files = [	'dynamic_agents_full_cooperation.txt',
			'dynamic_agents_partial_cooporation.txt',
			'dynamic_and_static_agents_large_space.txt',
			'dynamic_and_static_agents_narrow_space.txt'
		]
PdList = [0.5, 0.7, 0.9]
NList = [1, 3, 6]
lambdaPhiList = [0, 5e-5, 2e-4, 4e-4]
solvers = ["CPLEX"]#,"GLPK","CBC","GUROBI"]
nMonteCarlo = 15
threshold = 2 #meter

root = ET.Element("simulations", nMonteCarlo = str(nMonteCarlo))
for fileString in files:
	filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
	doc = ET.SubElement(root,"file", name = os.path.basename(fileString))
	trueTracks = openGroundTruth(filePath)
	trackLength = len(trueTracks[0])
	for solver in solvers:
		for P_d in PdList:
			prob = ET.SubElement(doc, "P_d", value = str(P_d))
			for N in NList:
				num = ET.SubElement(prob, "N", value = str(N))
				for lambda_phi in lambdaPhiList:
					lambdaPhi = ET.SubElement(num,"lambda_phi", value = '{:5.0e}'.format(lambda_phi))
					nTracks = 0
					nLostTracks = 0
					print('{:40s}'.format(os.path.splitext(fileString)[0]),solver,"P_d =",P_d,"N =",N,"lPhi =",'{:5.0e}'.format(lambda_phi), end = "\t")
					savefilePath = (os.path.join(loadLocation,os.path.splitext(fileString)[0],"results",os.path.splitext(fileString)[0])
										+"["
										+solver.upper()
										+",Pd="+str(P_d)
										+",N="+str(N)
										+",lPhi="+'{:7.5f}'.format(lambda_phi)
										+"]"
										+".xml")
					try:
						simulations = ET.parse(savefilePath).getroot()
					except FileNotFoundError:
						print("x"*nMonteCarlo)
						continue
					nSimulations = int(simulations.attrib.get('nMonteCarlo'))
					print("."*nSimulations,"x"*(nMonteCarlo-nSimulations),sep = "",end = "", flush = True)
					# for i in range(nMonteCarlo):	
					# 	estimatedTracks = openRes(savefilePath)
					# 	if estimatedTracks is None:
					# 		print("x",end = "")
					# 		continue
					# 	if trueTracks.shape != estimatedTracks.shape:
					# 		os.remove(savefile)
					# 		print("o",end = "")
					# 		continue
					# 	lostTracks = np.linalg.norm(trueTracks-estimatedTracks,2,2) > threshold
					# 	lostTracksTime = [np.flatnonzero(lostTrack).tolist() for lostTrack in lostTracks]
					# 	permanentLostTracks = []
					# 	for lostTrackTime in lostTracksTime:
					# 		if len(lostTrackTime):
					# 			permanentLostTracks.append(lostTrackTime[-1] == trackLength-1)
					# 	nLostTracks += sum(permanentLostTracks)
					# 	nTracks += len(estimatedTracks)
					# 	print(".", end = "")
					if nTracks != 0:
						print("\t",'{:3.0f}'.format(nLostTracks),"/",'{:3.0f}'.format(nTracks),"=>",'{:4.1f}'.format((nLostTracks/nTracks)*100),"%")
						ET.SubElement(lambdaPhi,"nTracks").text 	= str(nTracks)
						ET.SubElement(lambdaPhi,"nLostTracks").text = str(nLostTracks)
					else:
						print()

tree = ET.ElementTree(root)
tree.write("compareResult.xml")