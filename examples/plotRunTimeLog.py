#!/usr/bin/env python3
import os, sys
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
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
			figure = plt.figure(figsize = (12,16))
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
					xArray = np.array(lambdaList)
					if plotArray.any():
						nRow,nCol = plotArray.shape
						rectList = []
						for color, col in zip(['r','y','g','m'],range(nCol)):
							bottom = np.sum(plotArray[:,0:col], axis = 1)
							x = xArray - totalWidth/2 + j*width
							y = plotArray[:,col]
							# print(x,plotArray[:,col],y,z,bottom, color, sep = "\n", end = "\n\n")
							rects = plt.bar(x,y, width = width, bottom = bottom, color = color, alpha = 0.7, linewidth = 0)
							rectList.append(rects[0])
							plt.text(0.0006, 1,"N="+str(Nvalue))
							plt.xlabel("$\lambda_\phi$")
							plt.ylabel("Time (sec)")
							plt.xlim(0-totalWidth/2,0.0008+totalWidth/2)
						if (i == 0) and (j==0):
							plt.legend(rectList,categories,loc = 9, fontsize = 14, ncol = 4)
						for col in range(nRow):
							plt.text(xArray[col]-totalWidth/2+j*width+width/2,np.sum(plotArray[col,:])*0,"$P_D=$"+str(PdValue), 
								rotation = 'vertical',
								verticalalignment = 'bottom',
								horizontalalignment = "center",
								fontsize = 9)

			latexSaveFilePath = os.path.abspath(
								 	os.path.join("..","..","02 Latex","Figures",
									os.path.splitext(fileString)[0] +"-"+solverString+"_runtimeLog"+".pdf")
								 	)
			if not os.path.exists(os.path.dirname(latexSaveFilePath)):
						os.makedirs(os.path.dirname(latexSaveFilePath))
			print("Saving:", latexSaveFilePath)
			plt.tight_layout()
			# plt.show()
			figure.savefig(latexSaveFilePath, bbox_inches='tight')
			plt.close()
		# 	break
		# break
if __name__ == '__main__':
	os.chdir( os.path.dirname(os.path.abspath(__file__)) )
	plotRuntime()