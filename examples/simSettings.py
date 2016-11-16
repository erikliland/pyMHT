import os
loadLocation = os.path.join("..","data")
files 		= [	'dynamic_agents_full_cooperation.txt',
				'dynamic_agents_partial_cooporation.txt',
				'dynamic_and_static_agents_large_space.txt',
				'dynamic_and_static_agents_narrow_space.txt'
				]
PdList 		= [0.5, 0.7, 0.9]
NList 		= [1, 3, 6]
lambdaPhiList = [0, 5e-5, 1e-4, 2e-4]
solvers 	= ["CPLEX","GLPK","CBC","GUROBI"]
solvers.sort()
nMonteCarlo = 12
lambda_nu 	= 0.0001 #Expected number of new targets per unit volume 
sigma 		= 3		 #Need to be changed to conficence
threshold 	= 2 #meter

def generateFilePath(fileString, solver, P_d, N, lambda_phi):
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