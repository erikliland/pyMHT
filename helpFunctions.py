import matplotlib.pyplot as plt
import numpy as np

def plotVelocityArrowFromNode(nodes, stepsBack = 1):
	def recPlotVelocityArrowFromNode(node, stepsLeft):
		if node.predictedStateMean is not None:
			plotVelocityArrow(node)
		if stepsLeft > 0 and (node.parent is not None):
			recPlotVelocityArrowFromNode(node.parent, stepsLeft-1)
	for node in nodes:
		recPlotVelocityArrowFromNode(node, stepsBack)

def plotVelocityArrow(target):
	ax = plt.subplot(111)
	deltaPos = target.predictedStateMean[0:2] - target.initial.position.toarray()
	ax.arrow(target.initial.position.x, target.initial.position.y, deltaPos[0], deltaPos[1],
	head_width=0.1, head_length=0.1, fc= "None", ec='k', 
	length_includes_head = "true", linestyle = "-", alpha = 0.3, linewidth = 0.5)

def plotRadarOutline(centerPosition, radarRange):
	from matplotlib.patches import Ellipse
	plt.plot(centerPosition.x, centerPosition.y,"bo")
	ax = plt.subplot(111)
	circle = Ellipse(centerPosition.toarray(), radarRange*2, radarRange*2)
	circle.set_facecolor("none")
	circle.set_linestyle("dotted")
	ax.add_artist(circle)

def plotCovarianceEllipse(cov, Position, sigma):
	from matplotlib.patches import Ellipse
	lambda_, v = np.linalg.eig(cov)
	np.set_printoptions(precision = 3)
	ax = plt.subplot(111)
	ell = Ellipse( xy	 = (Position[0], Position[1]), 
				   width = np.sqrt(lambda_[0])*sigma*2, 
				   height= np.sqrt(lambda_[1])*sigma*2, 
				   angle = np.rad2deg( np.arctan2( lambda_[1], lambda_[0]) ))
	ell.set_facecolor('none')
	ell.set_linestyle("dotted")
	ell.set_alpha(0.3)
	ax.add_artist(ell)

def plotMeasurementList(measurmentList, scanIndex = None):
	for measurementIndex, measurement in enumerate(measurmentList.measurements):
		plotMeasurement(measurmentList, measurementIndex+1, scanIndex)

def plotMeasurementsFromList(scanHistory):
	for scanIndex, scan in enumerate(scanHistory):
		for measurementIndex, measurement in enumerate(scan.measurements):
			plotMeasurement(measurement, measurementIndex+1, scanIndex)

def plotMeasurementsFromForest(targetList, plotReal = True, plotDummy = True, **kwargs):
	def recPlotMeasurements(target, plottedMeasurements, plotReal, plotDummy):
		if target.parent is not None:
			if target.measurementNumber == 0:
				if plotDummy:
					plotMeasurement(target.initial.position, target.measurementNumber, target.scanIndex)
			else:
				if plotReal:
					measurementID = (target.scanIndex,target.measurementNumber)
					if measurementID not in plottedMeasurements:
						plotMeasurement(target.measurement, target.measurementNumber, target.scanIndex)
						plottedMeasurements.add( measurementID )
		for hyp in target.trackHypotheses:
			recPlotMeasurements(hyp, plottedMeasurements, plotReal, plotDummy)
	
	plotReal = kwargs.get('real', plotReal)
	plotDummy = kwargs.get('dummy', plotDummy)
	if not (plotReal or plotDummy):
		return
	plottedMeasurements = set()
	for target in targetList:
		recPlotMeasurements(target,plottedMeasurements,plotReal, plotDummy)

def plotMeasurement(position, measurementNumber = None, scanIndex = None):
	x = position.x
	y = position.y
	if measurementNumber == 0:
		plt.plot(x,y,color = "black",fillstyle = "none", marker = "o")
	else:
		plt.plot(x, y,'kx')
	if (scanIndex is not None) and (measurementNumber is not None):
		ax = plt.subplot(111)
		ax.text(x, y,str(scanIndex)+":"+str(measurementNumber), size = 7, ha = "left", va = "top") 

def plotValidationRegion(target,sigma, C, R):
	plotCovarianceEllipse(C.dot(target.predictedStateCovariance.dot(C.T))+R,
								C.dot(target.predictedStateMean), sigma)

def plotActiveTrack(associationHistory):
	def recBacktrackPosition(target):
		if target.parent is None:
			return [target.initial.position]
		return recBacktrackPosition(target.parent) + [target.initial.position] 

	for hyp in associationHistory:
		positions = recBacktrackPosition(hyp)
		plt.plot([p.x for p in positions], [p.y for p in positions])

def printScanList(scanList):
	for index, measurement in enumerate(scanList):
		print("\tMeasurement ", index, ":\t", end="", sep='')
		measurement.print()

def printClusterList(clusterList):
	print("Clusters:")
	for clusterIndex, cluster in enumerate(clusterList):
		print("Cluster ", clusterIndex, " contains target(s):\t", cluster, sep ="", end = "\n")
	print()

def printTargetList(__targetList__):
	print("TargetList:")
	for targetIndex, target in enumerate(__targetList__):
		print("\tTarget: ", str(targetIndex), "\t", repr(target),"\n", target, sep = "")
	print()

def printHypothesesScore(__targetList__):
	def recPrint(target, targetIndex):
		if len(target.trackHypotheses) == 0:
			pass
		else:
			for hyp in target.trackHypotheses:
				recPrint(hyp, targetIndex)
	for targetIndex, target in enumerate(__targetList__):
		print(	"\tTarget: ",targetIndex,
 				"\tInit",	target.initial.position,
 				"\tPred",	target.predictedPosition(),
 				"\tMeas",	target.measurement,sep = "")

def pol2cart(bearingDEG,distance):
	angleDEG = 90 - bearingDEG
	angleRAD = np.deg2rad(angleDEG)
	x = distance * np.cos(angleRAD)
	y = distance * np.sin(angleRAD)
	return [x,y]

def NLLR(hypothesisIndex,P_d, measurement = None,predictedMeasurement = None,lambda_ex = None,covariance = None):
	if hypothesisIndex == 0:
		return -np.log(1-P_d)
	else:
		measurementResidual = measurement.toarray() - predictedMeasurement
		return (	0.5*(measurementResidual.T.dot(np.linalg.inv(covariance)).dot(measurementResidual))
					+ np.log((lambda_ex*np.power(np.linalg.det(2*np.pi*covariance),0.5))/P_d) 	)

def backtrackMeasurementsIndices(selectedNodes):
	def recBacktrackNodeMeasurements(node, measurementBacktrack):
		if node.parent is not None:
			measurementBacktrack.append(node.measurementNumber)
			recBacktrackNodeMeasurements(node.parent,measurementBacktrack)
	measurementsBacktracks = []
	for leafNode in selectedNodes:
		measurementBacktrack = []
		recBacktrackNodeMeasurements(leafNode, measurementBacktrack)
		measurementBacktrack.reverse()
		measurementsBacktracks.append(measurementBacktrack)
	return measurementsBacktracks

# def _hypothesesIndices2measurementsIndices(measurementList, selectedHypotheses, A1):
# from operator import itemgetter
# numOfTimesteps = max(measurementList,key=itemgetter(1))[0]
# print("numOfTimesteps",numOfTimesteps)
# measurementsIndices = []
# for hypIndex in selectedHypotheses:
# 	hypMeasIndices = np.nonzero(A1[:,hypIndex])[0]
# 	print("hypMeasIndices", A1[:,hypIndex] )
# 	hypMeas = [measurementList[i] for i in hypMeasIndices]
# 	print("hypMeas",hypMeas)
# 	hypMeas.sort(key=lambda tup: tup[0])
# 	measurementsIndices.append( [meas[1] for meas in hypMeas] )
# return measurementsIndices