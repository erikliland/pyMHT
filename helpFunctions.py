def plotEstimatedPosition(state):
	from matplotlib.pyplot import plot
	plot(state[0], state[1], color = "black", fillstyle = "none", marker = "o")

def plotVelocityArrow(targetList, timeStep):
	from classDefinitions import Target
	from matplotlib.pyplot import arrow, subplot
	ax = subplot(111)
	for target in targetList:
		ax.arrow(target.position.x, target.position.y, target.velocity.x*timeStep, target.velocity.y*timeStep,
		head_width=0.1, head_length=0.1, fc= "None", ec='k', 
		length_includes_head = "true", linestyle = "-", alpha = 0.7, linewidth = 0.5)

def plotRadarOutline(centerPosition, radarRange):
	from classDefinitions import Position
	import matplotlib.pyplot as plt
	from matplotlib.patches import Ellipse
	plt.plot(centerPosition.x, centerPosition.y,"bo")
	ax = plt.subplot(111)
	circle = Ellipse(centerPosition.array(), radarRange*2, radarRange*2)
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
	#print("Cov:\n", cov[0:2,0:2])
	#print("Lambda:", lambda_)
	ax = plt.subplot(111)
	ell = Ellipse( xy	 = (Position[0], Position[1]), 
				   width = lambda_[0]*sigma*2, 
				   height= lambda_[1]*sigma*2, 
				   angle = rad2deg(arccos(v[0,0])))
	ell.set_facecolor('none')
	ell.set_linestyle("dotted")
	ax.add_artist(ell)

def plotTargetList(targetList):
	from matplotlib.pyplot import plot
	for target in targetList:
		plot([target.position.x],[target.position.y],"k+")

def plotMeasurements(measurmentList):
	import matplotlib.pyplot as plt
	#plotHandles = []
	ax = plt.subplot(111)
	for measurement in measurmentList.measurements:
		x = measurement.x
		y = measurement.y
		plt.plot(x, y,'kx')
		#ax.add_artist( plt.text(x,y-0.8,"K"+str(scanIndex)+"-"+str(targetIndex), size = 8, ha = "center", va = "top") )
		#tempPlot, = plt.plot(xList[1:], yList[1:],"x", label = "Target "+str(targetIndex))
		#plotHandles.append(tempPlot)

def printMeasurementMatrix(measurmentMatrix):
	for index, scanList in enumerate(measurmentMatrix):
		print("Scan:", index)
		printScanList(scanList)

def printScanList(scanList):
	for index, measurement in enumerate(scanList):
		print("\tMeasurement ", index, ":\t", end="", sep='')
		measurement.print()

def pol2cart(bearingDEG,distance):
	from numpy import deg2rad, cos, sin
	from classDefinitions import Position
	angleDEG = 90 - bearingDEG
	angleRAD = deg2rad(angleDEG)
	x = distance * cos(angleRAD)
	y = distance * sin(angleRAD)
	return [x,y]
