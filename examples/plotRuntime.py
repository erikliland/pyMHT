#!/usr/bin/env python3
import os, sys
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.ElementTree as ET

def plotRuntime():
	file = "compareResult.xml"
	simulations = ET.parse(file).getroot()
	colors = sns.color_palette(n_colors = 4)
	sns.set_style(style='white')
	lineStyle = ['-','-.','--','-']
	for file in simulations.findall("file"):
		fileString = file.attrib.get("name")
		figure = plt.figure(figsize = (16,10))
		maxTime = 0
		ax = figure.add_subplot(111,projection = '3d')
		for k, solver in enumerate(file.findall("Solver")):
			solverString = solver.attrib.get("name")	
			for j, P_d in enumerate(solver.findall("P_d")):
				PdValue = float(P_d.attrib.get("value"))
				plotPlane = []
				for i, N in enumerate(P_d.findall("N")):
					Nvalue = int(N.attrib.get("value"))
					lambdaPhiList = []
					for lambdaPhi in N.findall("lambda_phi"):
						lambdaValue = float(lambdaPhi.attrib.get("value"))
						totalSimTime= float(lambdaPhi.find("totalTime").text)
						nSimulations = float(lambdaPhi.find("nSimulations").text)
						timePerSimulation = totalSimTime/nSimulations
						maxTime = max(maxTime, timePerSimulation)
						lambdaPhiList.append([lambdaValue,timePerSimulation])
					plotArray = np.array(lambdaPhiList)
					if plotArray.any():
						x = plotArray[:,0]
						y = np.ones(len(lambdaPhiList))*PdValue*100
						z = (plotArray[:,1])
						ax.plot(x,y,z,lineStyle[i], label = "N="+str(Nvalue)+","+solverString if (j == 0) else None, c = colors[k], linewidth = 2)
#		ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.8), fontsize = 18)
		ax.legend(loc = 9, fontsize = 18, ncol = 4)
		ax.view_init(15, -163)
		ax.set_xlabel("\n$\lambda_{\phi}$", fontsize = 18, linespacing = 3)
		ax.set_zlabel("\nRuntime per simulation (sec)", fontsize = 18, linespacing = 3)
		ax.set_ylabel("\nProbability of detection (%)", fontsize = 18, linespacing = 2)
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
		ax.set_zlim(0,maxTime)
		#ax.set_zscale('log')
		ax.tick_params(labelsize = 16, pad = 1)
		yStart, yEnd = ax.get_ylim()
		ax.yaxis.set_ticks(np.arange(yStart, yEnd*1.1, 10))
		xStart, xEnd = ax.get_xlim()
		ax.xaxis.set_ticks(np.arange(xStart, xEnd*1.1, 1e-4))
		xTickLabels = ax.xaxis.get_ticklabels()
		for label in xTickLabels:
			label.set_verticalalignment('bottom')
			label.set_horizontalalignment('left')
			label.set_rotation(0)
		#plt.title(os.path.splitext(fileString)[0] + "-" + solverString)
		latexSaveFilePath = os.path.join("..","..","02 Latex","Figures",os.path.splitext(fileString)[0] +"_runtime"+".pdf")
		if not os.path.exists(os.path.dirname(latexSaveFilePath)):
					os.makedirs(os.path.dirname(latexSaveFilePath))
		figure.savefig(latexSaveFilePath, bbox_inches='tight')
		# plt.show()

if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	plotRuntime()