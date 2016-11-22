#!/usr/bin/env python3
import getopt
import sys
import os
import simSettings as s

def deleteFiles(**kwargs):
	from simSettings import generateFilePath
	loadLocation 	= kwargs.get("loadLocation",[])
	files 			= kwargs.get("files",[])
	solvers 		= kwargs.get("solvers",[])
	PdList 			= kwargs.get("PdList",[])
	NList 			= kwargs.get("NList",[])
	lambdaPhiList 	= kwargs.get("lambdaPhiList",[])
	for fileString in files:
		for solver in solvers:
			for P_d in PdList:
				for N in NList:
					for lambda_phi in lambdaPhiList:
						relativePath = s.generateResultFilePath(fileString, solver, P_d, N, lambda_phi)
						absPath = os.path.abspath(relativePath)
						try:
							if os.path.isfile(absPath):
								os.remove(absPath) 
								print("Removed", absPath)
						except OSError:
							print("Failed to remove",absPath)
							pass

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	confirmation = input("Are you sure you want to delete all those files? [Yes/No]")
	if confirmation == "Yes":
		deleteFiles(	loadLocation = s.loadLocation,
						files = s.croppedFiles,
						solvers = s.solvers,
						PdList = s.PdList,
						NList = s.NList,
						lambdaPhiList = s.lambdaPhiList
						)
		print("There you go! All these files are gone...")
	else:
		print("Phuuu! The files are still there... Smart choice!")