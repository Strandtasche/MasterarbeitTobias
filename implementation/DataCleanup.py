#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from loadDataExample import _removeNans
import matplotlib.pyplot as plt
from MaUtil import _distanceEu
import logging
import glob
import os
import sys
import time
from random import random
import weakref

def unit_vector(vector):
	"""helper function: Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" helper function: Returns the angle in radians between vectors 'v1' and 'v2'"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def filterTracksByBestFitStraightLine(inputFile):
	"""function to visualize the effect filtering tracks by Best fit straight line would have"""
	
	THRESHOLD = 3000000
	
	df = pd.read_csv(inputFile)
	
	numberTracks = (df.shape[1]) / 2
	print(int(numberTracks))
	
	assert numberTracks == int(numberTracks)
	
	input("press enter")

	

	plt.axis([100, 2250, 0, 1750])
	
	for i in range(int(numberTracks)):
		trackNo = int(''.join([s for s in df.iloc[:, 2*i].name if s.isdigit()]))

		a = df.iloc[:, (2 * i)].values
		b = df.iloc[:, (2 * i + 1)].values

		a = _removeNans(a)
		b = _removeNans(b)

		p, res, _, _, _ = np.polyfit(a, b, 2, full=True)
		
		if res > THRESHOLD:
			features = plt.plot(a, b, 'ro')
			print("track: {}, res: {}".format(trackNo, res))
			plt.pause(1)
			f = features.pop(0).remove()
		

	plt.show()
	
	plt.close()
	# Result: best fit straight line is not a sensible way to detect tracks with collisions

	# print(df)


def filterTracksByAngleDifference(dataFrame, THRESHOLD=0.3, display=True):
	"""function to visualize the effect filtering tracks by Angle Difference line would have
	returns a list of column indices that exceed the given THRESHOLD"""
	
	# THRESHOLD = 0.3
	
	df = dataFrame
	
	numberTracks = (df.shape[1]) / 2
	
	assert numberTracks == int(numberTracks)
	plt.axis([100, 2250, 0, 1750])

	dropIndices = []
	
	for i in range(int(numberTracks)):
		trackNo = int(''.join([s for s in df.iloc[:, 2*i].name if s.isdigit()]))
		
		a = df.iloc[:, (2 * i)].values
		b = df.iloc[:, (2 * i + 1)].values
		
		a = _removeNans(a)
		b = _removeNans(b)
		
		vecs = []
		
		if len(a) <= 3:
			logging.info("track: {} - not enough points".format(trackNo))
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)
		
			if display:
				features = plt.plot(a, b, 'ro')
				plt.pause(3)
				f = features.pop(0).remove()
			continue
		
		for j in range(len(a) - 1):
			vecs.append((a[j+1] - a[j], b[j+1] - b[j]))
		
		angles = []
		
		for k in range(len(vecs) - 1):
			angles.append(angle_between(vecs[k], vecs[k+1]))
			
			
		# print("track: {}, angles: {}".format(trackNo, max(angles)))
		
		if max(angles) > THRESHOLD:
			logging.info("track: {}, angles: {}".format(trackNo, max(angles)))
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)
			
			if display:
				features = plt.plot(a, b, 'ro')
				plt.pause(3)
				f = features.pop(0).remove()
	
	if display:
		plt.show()
	
		plt.close()
		
	logging.info("removed {}/{} tracks".format(int(len(dropIndices)/2), int(numberTracks)))
	
	return dropIndices


def filterTracksByVectorLength(dataFrame, THRESHOLD=10, display=True):
	
	# THRESHOLD = 10
	
	df = dataFrame
	
	numberTracks = (df.shape[1]) / 2
	
	assert numberTracks == int(numberTracks)
	plt.axis([100, 2250, 0, 1750])
	
	dropIndices = []
	
	for i in range(int(numberTracks)):
		trackNo = int(''.join([s for s in df.iloc[:, 2*i].name if s.isdigit()]))
		
		a = df.iloc[:, (2 * i)].values
		b = df.iloc[:, (2 * i + 1)].values
		
		a = _removeNans(a)
		b = _removeNans(b)
		
		vecs = []
		
		if len(a) <= 3:
			logging.info("track: {} - not enough points".format(trackNo))
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)
			
			if display:
				plt.axis([100, 2250, 0, 1750])
				plt.title("Track {}".format(trackNo))
				features = plt.plot(a, b, 'ro')
				plt.pause(3)
				f = features.pop(0).remove()
			continue
		
		for j in range(len(a) - 1):
			vecs.append((a[j+1] - a[j], b[j+1] - b[j]))
		
		lengths = [np.linalg.norm(i) for i in vecs]
		lenDiffs = np.diff(lengths)
		# continue
		
		# print("track: {}, angles: {}".format(trackNo, max(angles)))
	
		# print("max difference: {}".format(max(lengths) - min(lengths)))
		
		if (max(lenDiffs)) > THRESHOLD:
			logging.info("track: {}, lendif: {}".format(trackNo, max(lenDiffs)))
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)

			if display:
				plt.axis([100, 2250, 0, 1750])
				plt.title("Track {}".format(trackNo))
				features = plt.plot(a, b, 'ro')
				plt.pause(3)
				f = features.pop(0).remove()
	
	if display:
		plt.show()
		
		plt.close()
	
	logging.info("removed {}/{} tracks".format(int(len(dropIndices)/2), int(numberTracks)))
	
	return dropIndices


def cleanUpFolder(path, target):
	""" cleanup function that drops tracks determined to be collision and writes new csv files to some location"""

	folder = path
	
	fileList = []
	if folder[-4:] != '.csv':
		fileList = sorted(glob.glob(folder + '/*.csv'))
		logging.info("getting all csv files in {}".format(folder))
	else:
		fileList.append(folder)
		logging.info("loading file {}".format(folder))
	
	assert len(fileList) > 0, "no files found input location " + folder
	dataFrameList = []
	
	if not (os.path.exists(target) or os.access(os.path.dirname(target), os.W_OK)):
		# invalid
		logging.error("invalid target")
		sys.exit(-1)
	# else: # implicit else
	
	for elem in fileList:
		basename = os.path.basename(elem)
		logging.info("Write to " + target + "/" + basename )
		df = pd.read_csv(elem)
		dropIndicesAngle = filterTracksByAngleDifference(df, THRESHOLD=0.3, display=False)
		df.drop(df.columns[dropIndicesAngle], axis=1, inplace=True)
		dropIndicesLength = filterTracksByVectorLength(df, THRESHOLD=10, display=False)
		df.drop(df.columns[dropIndicesLength], axis=1, inplace=True)
		df.to_csv(os.path.join(target, basename), index=False, na_rep="NaN", encoding='utf-8')

