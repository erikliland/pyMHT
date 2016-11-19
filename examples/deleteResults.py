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
						savefilePath = generateFilePath(fileString, solver, P_d, N, lambda_phi)
						try:
							os.remove(savefilePath) 
							print("Removing", savefilePath)
						except OSError:
							pass

if __name__ == '__main__':
	confirmation = input("Are you sure you want to delete all those files? [Yes/No]")
	if confirmation == "Yes":
		deleteFiles(	loadLocation = s.loadLocation,
						files = s.croppedFiles,
						solvers = s.solvers,
						PdList = s.PdList,
						NList = s.NList,
						lambdaPhiList = s.lambdaPhiList
						)
	else:
		print("Phuuu! The files are still there... Smart choice!")