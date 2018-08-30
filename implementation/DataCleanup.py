#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from loadDataExample import _removeNans
import matplotlib.pyplot as plt
import time
from random import random
import weakref

def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'::
	"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def filterTracksByBestFitStraightLine(inputFile):
	
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


	# print(df)


def filterTracksByAngleDifference(inputFile):
	
	THRESHOLD = 0.3
	
	df = pd.read_csv(inputFile)
	
	numberTracks = (df.shape[1]) / 2
	print(int(numberTracks))
	
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
			print("track: {} - not enough points".format(trackNo))
			
			features = plt.plot(a, b, 'ro')
			plt.pause(3)
			f = features.pop(0).remove()
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)
			continue
		
		for j in range(len(a) - 1):
			vecs.append((a[j+1] - a[j], b[j+1] - b[j]))
		
		angles = []
		
		for k in range(len(vecs) - 1):
			angles.append(angle_between(vecs[k], vecs[k+1]))
			
			
		# print("track: {}, angles: {}".format(trackNo, max(angles)))
		
		if max(angles) > THRESHOLD:
			features = plt.plot(a, b, 'ro')
			print("track: {}, angles: {}".format(trackNo, max(angles)))
			plt.pause(3)
			f = features.pop(0).remove()
			dropIndices.append(2*i)
			dropIndices.append(2*i + 1)
	
	
	plt.show()
	
	plt.close()
	
	return dropIndices


