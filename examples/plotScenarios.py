#!/usr/bin/env python3
import os
import sys
import signal 
import time
import functools
import pulp
import argparse

import xml.etree.ElementTree as ET
import multiprocessing as mp 
import matplotlib.pyplot as plt

import tomht
import tomht.radarSimulator as sim
import tomht.helpFunctions as hpf
import tomht.stateSpace.pv as model
from simSettings import *

def plotScenarios():
	fig1 = plt.figure(num=1, figsize = (9,9), dpi=100)	
	for index, fileString in enumerate(croppedFiles):
		filePath = os.path.join(loadLocation,os.path.splitext(fileString)[0],fileString)
		(initialTargets, simList) = sim.importFromFile(filePath)
		(p0, radarRange) = sim.findCenterPositionAndRange(simList)

		tracker = tomht.Tracker(model.Phi, model.C, model.Gamma, 1, model.P0, model.R(), model.Q, 0, 0, eta2, 1, 0, "CBC", logTime = True)
		for initialTarget in initialTargets:
		 	tracker.initiateTarget(initialTarget)
		fig1.clf()
		hpf.plotTrueTrack(simList, alpha = 0.1)
		tracker.plotInitialTargets()
		plt.axis("equal")
		plt.xlim((p0.x-radarRange*1.05, p0.x + radarRange*1.05))
		plt.ylim((p0.y-radarRange*1.05, p0.y + radarRange*1.05))
		fig1.canvas.draw()
		fig1.savefig(os.path.join("..","..","02 Latex","Figures","scenario"+str(index)+".png"), bbox_inches='tight')

if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	plotScenarios()