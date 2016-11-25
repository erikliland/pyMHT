#!/usr/bin/env python3
import os


def generateResultFilePath(fileString, solver, P_d, N, lambda_phi):
	return (
		os.path.join(	loadLocation,
						os.path.splitext(fileString)[0],
						"results",
						os.path.splitext(fileString)[0]
						) 	+"["
							+solver.upper()
							+",Pd="+str(P_d)
							+",N="+str(N)
							+",lPhi="+'{:7.5f}'.format(lambda_phi)
							+"]"
							+".xml"
			)

def generateRawFileName(fileName):
	return os.path.splitext(fileName)[0]+"-RAW.txt"

def generateRawFilePath(fileName, rawFileName):
	return os.path.join(loadLocation,os.path.splitext(fileName)[0],rawFileName)

def generateFilePath(fileName):
	return os.path.join(loadLocation,os.path.splitext(fileName)[0],fileName)

def generateCroppedFileName(fileName):
	return os.path.splitext(os.path.basename(fileName))[0]+"_cropped.txt"

def generateCroppedFilePath(croppedFileName):
	return os.path.join(loadLocation,os.path.splitext(croppedFileName)[0], croppedFileName)

loadLocation = os.path.join("..","data")
files 		= [	'dynamic_agents_full_cooperation.txt',
				'dynamic_agents_partial_cooporation.txt',
				'dynamic_and_static_agents_large_space.txt',
				'dynamic_and_static_agents_narrow_space.txt',
				'parallel_targets.txt'
				]
rawFiles 	= [generateRawFileName(file) for file in files]
croppedFiles= [generateCroppedFileName(file) for file in files]
PdList 		= [0.5, 0.6, 0.7, 0.8, 0.9]
NList 		= [1, 3, 6]
lambdaPhiList = [0, 1e-4, 2e-4, 4e-4, 8e-4]
solvers 	= ["CBC","CPLEX","GLPK","GUROBI"]
nMonteCarlo = 10	#Number of copies of each simulation (random)
lambda_nu 	= 0.0001#Expected number of new targets per unit volume 
# etaconfidence 	= 0.95	#Chi2 inverse cdf, df=2
eta2 		= 5.99 #chi2.ppf(confidence,2)
threshold 	= 4 	#meter
