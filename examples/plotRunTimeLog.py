#!/usr/bin/env python3
import os, sys
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ast
import xml.etree.ElementTree as ET

def plotRuntime():
	file = "compareResult.xml"
	simulations = ET.parse(file).getroot()
	totalWidth = 0.00008
	categories = ['Process', 'Cluster','Optim','Prune']
	sns.set_style(style='white')
	lineStyle = ['-','-.','--','-','-']
	for file in simulations.findall("file"):
		fileString = file.attrib.get("name")
		for k, solver in enumerate(file.findall("Solver")):
			solverString = solver.attrib.get("name")
			figure = plt.figure(figsize = (10,10))
			# ax = figure.add_subplot(111,projection = '3d')
			for j, P_d in enumerate(solver.findall("P_d")):
				PdValue = float(P_d.attrib.get("value"))
				plotPlane = []
				allN = P_d.findall("N")
				nN = len(allN)
				width = totalWidth/nN
				for i, N in enumerate(allN):
					plt.subplot(nN,1,i+1)
					Nvalue = int(N.attrib.get("value"))
					lambdaList = []
					runTimeList = []
					for lambdaPhi in N.findall("lambda_phi"):
						lambdaValue = float(lambdaPhi.attrib.get("value"))
						lambdaList.append(lambdaValue)
						runTimeLogAvg= ast.literal_eval(lambdaPhi.find("runTimeLogAvg").text)
						tempList = []
						for k in categories:
							tempList.append(runTimeLogAvg.get(k,0))
						runTimeList.append(tempList)
					plotArray = np.array(runTimeList)
					if plotArray.any():
						nRow,nCol = plotArray.shape
						for color, col in zip(['r','c','g','b'],range(nCol)):
							bottom = np.sum(plotArray[:,0:col], axis = 1)
							x = np.array(lambdaList) - totalWidth/2 + j*width
							y = plotArray[:,col]
							# z = np.ones(len(runTimeList))*PdValue*100
							# print(x,plotArray[:,col],y,z,bottom, color, sep = "\n", end = "\n\n")
							plt.bar(x,y, width = width, bottom = bottom, color = color, alpha = 0.7)#label = categories[col] if (PdValue == 0.9)and(lambdaValue==0) else None
							plt.text(0.0006, 1,"N="+str(Nvalue))
							plt.xlabel("$\lambda_\phi$")
							plt.ylabel("Time (sec)")
							plt.xlim(0-totalWidth/2,0.0008+totalWidth/2)
	#		ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.8), fontsize = 18)
			# ax.legend(loc = 9, fontsize = 18, ncol = 4)
			# ax.view_init(6, -151)
			# ax.set_xlabel("\n$\lambda_{\phi}$", fontsize = 18, linespacing = 3)
			# ax.set_zlabel("\nRuntimelog per simulation (sec)", fontsize = 18, linespacing = 3)
			# ax.set_ylabel("\nProbability of detection (%)", fontsize = 18, linespacing = 2)
			# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
			# ax.tick_params(labelsize = 16, pad = 1)
			# ax.set_zlim(0,11)
			# yStart, yEnd = ax.get_ylim()
			# ax.yaxis.set_ticks(np.arange(yStart, yEnd*1.1, 10))
			# xStart, xEnd = ax.get_xlim()
			# xTickLabels = ax.xaxis.get_ticklabels()
			# for label in xTickLabels:
			# 	label.set_verticalalignment('bottom')
			# 	label.set_horizontalalignment('left')
			# 	label.set_rotation(0)
			latexSaveFilePath = os.path.abspath(
								 	os.path.join("..","..","02 Latex","Figures",
									os.path.splitext(fileString)[0] +"-"+solverString+"_runtimeLog"+".pdf")
								 	)
			if not os.path.exists(os.path.dirname(latexSaveFilePath)):
						os.makedirs(os.path.dirname(latexSaveFilePath))
			print("Saving:", latexSaveFilePath)
			plt.show()
			break
		break
			# figure.savefig(latexSaveFilePath, bbox_inches='tight')
			# plt.close()

if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	plotRuntime()