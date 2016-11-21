#!/usr/bin/env python3
import os, sys
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.ElementTree as ET

def plotResults():
	file = "compareResult.xml"
	simulations = ET.parse(file).getroot()
	colors = sns.color_palette(n_colors = 4)
	sns.set_style(style='white')
	for file in simulations.findall("file"):
		fileString = file.attrib.get("name")
		for solver in file.findall("Solver"):
			solverString = solver.attrib.get("name")
			figure = plt.figure(figsize = (10,10), dpi = 100)
			ax = figure.add_subplot(111,projection = '3d')
			for j, P_d in enumerate(solver.findall("P_d")):
				PdValue = float(P_d.attrib.get("value"))
				plotPlane = []
				for i, N in enumerate(P_d.findall("N")):
					Nvalue = int(N.attrib.get("value"))
					lambdaPhiList = []
					for lambdaPhi in N.findall("lambda_phi"):
						lambdaValue = float(lambdaPhi.attrib.get("value"))
						# print(fileString, solverString, PdValue, Nvalue,lambdaValue)
						nTracks = float(lambdaPhi.find("nTracks").text)
						nLostTracks = float(lambdaPhi.find("nLostTracks").text)
						lambdaPhiList.append([lambdaValue,nLostTracks/nTracks])
					plotArray = np.array(lambdaPhiList)
					if plotArray.any():
						x = plotArray[:,0]
						y = np.ones(len(lambdaPhiList))*PdValue*100
						z = plotArray[:,1]#*np.random.normal(loc = 1, scale = 0.5)
						ax.plot(x,y,z,'--', label = "N="+str(Nvalue) if j == 0 else None, c = colors[i])
						# ax.plot(np.flipud(x),np.flipud(y),np.flipud(z),'o',c = colors[i], alpha = 0.7)
						# print(x,y,z)
						# print(np.flipud(x),np.flipud(y),np.flipud(z))
						
			ax.legend()
			ax.view_init(15, -150)
			ax.set_xlabel("$\lambda_{\phi}$")
			ax.set_zlabel("Track loss (%)")
			ax.set_ylabel("Probability of detection (%)")
			ax.ticklabel_format(style = "sci", axsis = "x", scilimits=(0,0))
			ax.set_zlim(0,1)
			plt.title(os.path.splitext(fileString)[0] + "-" + solverString)
			savefilePath = os.path.join("plots",os.path.splitext(fileString)[0] + "-" + solverString+".png")
			if not os.path.exists(os.path.dirname(savefilePath)):
						os.makedirs(os.path.dirname(savefilePath))
			figure.savefig(savefilePath, bbox_inches='tight')
			# plt.show()

if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	plotResults()