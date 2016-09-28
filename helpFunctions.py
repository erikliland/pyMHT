def plotInitialTargetIndex(initialTarget, index):
	import matplotlib.pyplot as plt
	from numpy.linalg import norm
	ax = plt.subplot(111)
	normVelocity = initialTarget.velocity.toarray() / norm(initialTarget.velocity.toarray())
	offset = 0.2 * normVelocity
	position = initialTarget.position.toarray() - offset
	ax.text(position[0], position[1], index, 
		fontsize=8, horizontalalignment = "center", verticalalignment = "center")

def plotDummyMeasurement(target):
	from matplotlib.pyplot import plot
	plot(target.predictedStateMean[0], target.predictedStateMean[1], color = "black", fillstyle = "none", marker = "o")

def plotVelocityArrow(target):
	from matplotlib.pyplot import arrow, subplot
	ax = subplot(111)
	deltaPos = target.predictedStateMean[0:2] - target.initial.position.toarray()
	ax.arrow(target.initial.position.x, target.initial.position.y, deltaPos[0], deltaPos[1],
	head_width=0.1, head_length=0.1, fc= "None", ec='k', 
	length_includes_head = "true", linestyle = "-", alpha = 0.3, linewidth = 0.5)

def plotRadarOutline(centerPosition, radarRange):
	from classDefinitions import Position
	import matplotlib.pyplot as plt
	from matplotlib.patches import Ellipse
	plt.plot(centerPosition.x, centerPosition.y,"bo")
	ax = plt.subplot(111)
	circle = Ellipse(centerPosition.toarray(), radarRange*2, radarRange*2)
	circle.set_facecolor("none")
	circle.set_linestyle("dotted")
	ax.add_artist(circle)

def plotCovarianceEllipse(cov, Position, sigma):
	from numpy.linalg import eig
	from numpy import sqrt, rad2deg, arccos
	from matplotlib.patches import Ellipse
	import matplotlib.pyplot as plt

	lambda_, v = eig(cov[0:2,0:2])
	lambda_ = sqrt(lambda_)
	# print("Cov:\n", cov[0:2,0:2])
	# print("Lambda:", lambda_)
	# print("Sigma:", sigma)
	ax = plt.subplot(111)
	ell = Ellipse( xy	 = (Position[0], Position[1]), 
				   width = lambda_[0]*sigma*2, 
				   height= lambda_[1]*sigma*2, 
				   angle = rad2deg(arccos(v[0,0])))
	ell.set_facecolor('none')
	ell.set_linestyle("dotted")
	ell.set_alpha(0.3)
	ax.add_artist(ell)

def plotMeasurements(measurmentList):
	import matplotlib.pyplot as plt
	ax = plt.subplot(111)
	for measurement in measurmentList.measurements:
		x = measurement.x
		y = measurement.y
		plt.plot(x, y,'kx')

def plotDummyMeasurementIndex(scanIndex, target):
	import matplotlib.pyplot as plt
	ax = plt.subplot(111)
	ax.text(target.predictedStateMean[0],
			target.predictedStateMean[1],
			str(scanIndex)+":"+str(0), 
			size = 7, ha = "left", va = "top") 

def plotMeasurementIndecies(scanIndex, measurements):
	import matplotlib.pyplot as plt
	ax = plt.subplot(111)
	for measurementIndex, measurement in enumerate(measurements):
		ax.text(measurement.x,
				measurement.y,
				str(scanIndex)+":"+str(measurementIndex+1), 
				size = 7, ha = "left", va = "top") 


def plotTargetList(targetList):
	from matplotlib.pyplot import plot
	for target in targetList:
		plot([target.position.x],[target.position.y],"k+")

def plotCovariance(target,sigma, C):
		from numpy import dot
		plotCovarianceEllipse(target.predictedStateCovariance, dot(C,target.predictedStateMean), sigma)

def printScanList(scanList):
	for index, measurement in enumerate(scanList):
		print("\tMeasurement ", index, ":\t", end="", sep='')
		measurement.print()

def printMeasurementAssociation(targetIndex, target):
	print("Track-measurement assosiation:")
	print("Target: ", targetIndex, "\n", target, sep = "")

def printClusterList(clusterList):
	print("Clusters:")
	for clusterIndex, cluster in enumerate(clusterList):
		print("Cluster ", clusterIndex, " contains target(s):\t", cluster, sep ="", end = "\n")
	print()

def printTargetList():
	print("TargetList:")
	for targetIndex, target in enumerate(__targetList__):
		print("Target: ", str(targetIndex), "\t",repr(target), sep = "")
	print()

def pol2cart(bearingDEG,distance):
	from numpy import deg2rad, cos, sin
	from classDefinitions import Position
	angleDEG = 90 - bearingDEG
	angleRAD = deg2rad(angleDEG)
	x = distance * cos(angleRAD)
	y = distance * sin(angleRAD)
	return [x,y]

