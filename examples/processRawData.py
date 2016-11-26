#!/usr/bin/env python3
import os
import sys
import argparse
import simSettings as sim

def cropRawFile(fileName,**kwargs):
	rawFileName 		= sim.generateRawFileName(fileName)
	loadRawFilePath 	= sim.generateRawFilePath(fileName,rawFileName) 
	saveFileName 		= fileName
	saveFilePath 		= sim.generateFilePath(saveFileName)
	saveCroppedFileName = sim.generateCroppedFileName(fileName) 
	saveCroppedFilePath = sim.generateCroppedFilePath(saveCroppedFileName)	
	print("Loading", rawFileName, 	  "  \t from \t", loadRawFilePath)
	print("Saving", saveFileName, 	"\t\t\t to   \t",saveFilePath)
	print("Saving", saveCroppedFileName,"\t t0   \t",saveCroppedFilePath)
	
	if not os.path.exists(os.path.dirname(saveFilePath)):
			os.makedirs(os.path.dirname(saveFilePath))
	if not os.path.exists(os.path.dirname(saveCroppedFilePath)):
			os.makedirs(os.path.dirname(saveCroppedFilePath))
	fIn = open(loadRawFilePath,'r')
	fOut = open(saveFilePath,'w')
	fOutCropped = open(saveCroppedFilePath,'w')
	for line in fIn:
		lineTime = float(line.split(',')[0])
		targets = line.strip().split(',')[1:]
		targets = targets[0:2*kwargs.get("maxTargets", len(targets)/2)]
		fOut.write( str( lineTime/kwargs.get("timeScale",1)) +"," + ",".join(targets)+"\n" )
		if lineTime >= kwargs.get("newStart",0):
			if lineTime <= kwargs.get("newEnd",float('inf')):
				fOutCropped.write( str((lineTime-kwargs.get("newStart",0))/kwargs.get("timeScale",1)) +"," + ",".join(targets)+"\n" )
	fIn.close()
	fOut.close()
	fOutCropped.close()


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	for file in sim.files[0:4]:
		cropRawFile(file, newStart = 100, timeScale = 2, newEnd = 200, maxTargets = 5)