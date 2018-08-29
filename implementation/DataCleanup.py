#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from loadDataExample import _removeNans
import matplotlib.pyplot as plt
import time
from random import random
import weakref

def filterTracks(inputFile):
	
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
