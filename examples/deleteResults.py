#!/usr/bin/env python3
import getopt
import sys
import os
import argparse
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
								# os.remove(absPath) 
								print("Removed", absPath)
						except OSError:
							print("Failed to remove",absPath)
							pass

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	parser = argparse.ArgumentParser(description = "Delete MHT tracker simulations", argument_default=argparse.SUPPRESS)
	parser.add_argument('-f', help = "Scenario number to delete", 	nargs = '+', type = int )
	parser.add_argument('-s', help = "Solver for ILP problem to delete",nargs = '+')
	parser.add_argument('-p', help = "Probability of detection", 	nargs = '+', type = float)
	parser.add_argument('-n', help = "Number of steps to remember",	nargs = '+', type = int )
	parser.add_argument('-l', help = "Lambda_Phi (clutter)", 		nargs = '+', type = float)
	args = vars(parser.parse_args())
	confirmation = input("Are you sure you want to delete all those files? [Yes/No]")
	# fileIndex = args.get("f")
	# if fileIndex is not None:
	# 	files = [sim.simFiles[i] for i in fileIndex]
	# else:
	# 	files 	= sim.simFiles
	if confirmation == "Yes":
		deleteFiles(	loadLocation = s.loadLocation,
						files = s.simFiles[args.get("f",0)],
						solvers = args.get("s",s.solvers),
						PdList = args.get("p",s.PdList),
						NList = args.get("n",s.NList),
						lambdaPhiList = args.get("l",s.lambdaPhiList)
						)
		print("There you go! All these files are gone...")
	else:
		print("Phuuu! The files are still there... Smart choice!")