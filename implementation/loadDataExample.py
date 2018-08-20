from __future__ import division

import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import logging
import glob
import bisect
import sys


# featureSize = 5

# CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5']

# CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4']

def genColumnNames(featureSize):
	"""returns an array that can be used as the name for feature columns"""
	names = []
	for i in range(featureSize):
		names.append('X_' + str(i))
	for i in range(featureSize):
		names.append('Y_' + str(i))

	return names


def loadCSVData(file='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv',
                featureSize=5):
	"""loads data from from CSV file and returns a dictionary with features and labels"""
	df = pd.read_csv(file)

	data = {"features": [], "labels": []}

	numberTracks = (df.shape[1] - 2) / 2

	for k in range(50):
		for i in range(int(numberTracks)):
			a = df.iloc[:, (2 + 2 * i)].values
			b = df.iloc[:, (2 + 2 * i + 1)].values
			sp = random.randint(0, df.shape[0] - featureSize - 2)
			final_feature = np.append(a[sp:(sp + featureSize)], b[sp:(sp + featureSize)])
			final_label = np.append(a[sp + featureSize + 1], b[sp + featureSize + 1])

			# assert final_feature.shape[0] == final_label.shape[0]

			data["features"].append(final_feature)
			data["labels"].append(final_label)

	return data


def loadCSVtoNpy(inputFile='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv',
                 outputfile='in.npy', featureSize=5):
	"""loads data from a csv file and generates a set of features and labels from it.
	the result is saved into a npy file and placed in the current folder"""

	dt = np.dtype([('features', float, (2 * featureSize,)), ('labels', float, (2,))])

	df = pd.read_csv(inputFile)
	numberTracks = (df.shape[1] - 2) / 2

	assert numberTracks == int(numberTracks)

	numberTracks = int(numberTracks)

	sampleNumberPerTrack = 5

	featureCount = numberTracks * sampleNumberPerTrack

	x = np.zeros((featureCount,), dtype=dt)

	counter = 0

	for i in range(numberTracks):
		a = df.iloc[:, (2 + 2 * i)].values
		b = df.iloc[:, (2 + 2 * i + 1)].values
		for k in range(sampleNumberPerTrack):
			sp = random.randint(0, df.shape[0] - featureSize - 2)
			x[counter]['features'] = np.append(a[sp:(sp + featureSize)], b[sp:(sp + featureSize)])
			x[counter]['labels'] = np.append(a[sp + featureSize + 1],
			                                 b[sp + featureSize + 1])  # a[sp + featureSize + 1]
			counter = counter + 1

	np.save(outputfile, x)


def _validateDF(dataFrame, featureSize=5):
	"""Private function: remove colums from dataframe which have too few datapoints to generate """

	df = dataFrame
	threshold = featureSize + 1  # has to fit at least 1 datapoint
	df.dropna(axis=1, thresh=threshold, inplace=True)

	# TODO: zusammenhängende werte?

	return df


def _removeNans(array):
	arr1d = array
	notnans = np.flatnonzero(~np.isnan(arr1d))
	if notnans.size:
		trimmed = arr1d[notnans[0]: notnans[-1] + 1]  # slice from first not-nan to the last one
	else:
		trimmed = np.zeros(0)

	return trimmed


# kopiert aus Stackoverflow thread "https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python"f
def _line(p1, p2):
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C


def _intersection(L1, L2):
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0:
		x = Dx / D
		y = Dy / D
		return x,y
	else:
		return False


def _distanceEu(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def prepareRawMeasNextStep(inputFile, featureSize=5):
	"""loads real(!) data from a csv file return a dataframe"""

	# dt = np.dtype([('features', float, (2 * featureSize,)), ('labels', float, (2,))])

	logging.info("Preparing Data from " + inputFile)

	df = pd.read_csv(inputFile)
	df = _validateDF(df, featureSize)

	numberTracks = (df.shape[1]) / 2 #TODO: Validate change

	assert numberTracks == int(numberTracks)

	numberTracks = int(numberTracks)

	columnNames = genColumnNames(featureSize)
	columnNames.append('LabelX')
	columnNames.append('LabelY')

	data = []

	for i in range(numberTracks):
		a = df.iloc[:, (2 * i)].values
		b = df.iloc[:, (2 * i + 1)].values

		# a = a[~np.isnan(a)]   #Remove nans from columns
		# b = b[~np.isnan(b)]   #Remove NaNs from Columns

		a = _removeNans(a)
		b = _removeNans(b)

		assert len(a) == len(b)

		featureCount = len(a) - featureSize
		for k in range(featureCount):
			temp = {}
			temp['features'] = np.append(a[k:(k+ featureSize)], b[k:(k+featureSize)])
			temp['labels'] = np.append(a[k + featureSize], b[k+featureSize])
			data.append(temp)

	# newDf = pd.DataFrame(columns=columNames)
	dataFrameList = []

	for elem in data:
		tempdict = {columnNames[k]: elem['features'][k] for k in range(2 * featureSize)}
		for i in [-2, -1]:
			tempdict[columnNames[i]] = elem['labels'][i]

		dataFrameList.append(pd.DataFrame(tempdict, index=[0]))


	# assert len(dataFrameList) > 0
	if len(dataFrameList) != 0:
		newDf = pd.concat(dataFrameList, ignore_index=True)

		sizeOld = newDf.shape[0]

		newDf.dropna(subset=['LabelX', 'LabelY'], inplace=True)

		if sizeOld != newDf.shape[0]:
			logging.info("removed(s) Row for Label NaN")
	
		sizeOld = newDf.shape[0]
		newDf.dropna(axis=0, inplace=True)
		
		if sizeOld != newDf.shape[0]:
			logging.info("removed Row(s) for Feature NaN")

	else:
		newDf = pd.DataFrame()
		logging.warning("no data found in " + inputFile)

	return newDf


def loadRawMeasNextStep(input, featureSize=5, testSize=0.1):
	"""loads all the raw measurement data from input into a pandas Dataframe and """

	# if input[:5] == "/home":
	# 	folder = input
	# else:
	# 	folder = '/home/hornberger/Projects/MasterarbeitTobias/' + input

	folder = input

	# TODO: handle single file inputs? os.path.isDir() - make system agnostic

	fileList = []
	if folder[-4:] != '.csv':
		fileList = sorted(glob.glob(folder + '/*.csv'))
		logging.info("getting all csv files in {}".format(folder))
	else:
		fileList.append(folder)
		logging.info("loading file {}".format(folder))

	assert len(fileList) > 0, "no files found input location " + folder
	dataFrameList = []

	for elem in fileList:
		dataFrameList.append(prepareRawMeasNextStep(elem, featureSize))

	newDf = pd.concat(dataFrameList, ignore_index=True)

	# Remove invalid rows: TODO: more sophisticated approach (forward/backwards fill?)

	if not pd.notnull(newDf).all().all():
		raise ValueError("NaNs in labels or feature break Neural Network")

	labelDf = newDf[['LabelX', 'LabelY']].copy()
	featureDf = newDf.drop(['LabelX', 'LabelY'], axis=1)

	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(featureDf, labelDf, test_size=testSize)

	return (trainFeatures, trainLabels), (testFeatures, testLabels)


def loadData(featureSize=5):
	"""DEPRECATED! use loadRawMeas() or loadFakeDataPandas() instead
	generates data in the right format from csv files and returns a
	dataset as (train_features, train_labels), (test_features, test_labels)."""

	prefix = os.getcwd()

	training_file = prefix + '/../data/GroundTruthExample/test_case1_labels.csv'
	test_file = prefix + '/../data/GroundTruthExample/test_case2_labels_NoNan.csv'

	trainingInNpFile = prefix + '/../data/GroundTruthExample/trainingIn.npy'
	testInNpFile = prefix + '/../data/GroundTruthExample/testIn.npy'

	loadCSVtoNpy(training_file, trainingInNpFile, featureSize)
	loadCSVtoNpy(test_file, testInNpFile, featureSize)

	train_data = np.load(trainingInNpFile)
	train_features = train_data["features"]
	train_labels = train_data["labels"]

	test_data = np.load(testInNpFile)
	test_features = test_data["features"]
	test_labels = test_data["labels"]

	CsvColumnNames = genColumnNames(featureSize)

	trainFeatureDict = {CsvColumnNames[k]: train_features[:, k] for k in range(2 * featureSize)}
	testFeatureDict = {CsvColumnNames[k]: test_features[:, k] for k in range(2 * featureSize)}

	# labelNames = ['X_n', 'Y_n']

	# trainLabelDict = {labelNames[k]: train_labels[:,k] for k in range(2)}
	# testLabelDict = {labelNames[k]: test_labels[:,k] for k in range(2)}

	assert train_features.shape[0] == train_labels.shape[0]
	assert test_features.shape[0] == test_labels.shape[0]

	return (trainFeatureDict, train_labels), (testFeatureDict, test_labels)


def prepareFakeData(startpX=700, startpY=0, veloX=-5, veloY=30, numberEx=75, featureSize=5):
	"""generate a single line and slice it up into as many examples as necessary. Return the examples as an array"""

	startpointx = startpX
	startpointy = startpY

	numberExamples = numberEx

	speedx = veloX
	speedy = veloY

	posX = [startpointx + i * speedx for i in range(numberExamples)]
	posY = [startpointy + i * speedy for i in range(numberExamples)]

	featureCount = numberExamples - featureSize

	assert numberExamples > (featureSize + 1) and featureSize > 0

	dt = np.dtype([('features', float, (2 * featureSize,)), ('labels', float, (2,))])
	data = np.zeros((featureCount,), dtype=dt)

	# print("posX: " + str(len(posX)))
	# print("feature count: " + str(featureCount))

	for i in range(featureCount):
		# print("Start: %d", i)
		# print("stop: %d", (i + featureSize))
		assert (i + featureSize) < numberExamples
		data[i]['features'] = np.append(posX[i:(i + featureSize)], posY[i:(i + featureSize)])
		data[i]['labels'] = np.append(posX[i + featureSize], posY[i + featureSize])

	return data


def loadFakeData(featureSize=5, numberOfLines=100, testSize=0.1):
	"""DEPRECATED: Use loadFakeDataPandas instead
	loads generated fake data into training and test tuples """
	dataArray = []

	# for i in range(numberOfLines):
	while len(dataArray) < numberOfLines:
		dataPoint = prepareFakeData(random.randint(500, 1200), 10 * random.randint(0, 15),
		                            random.randint(-15, 15), random.uniform(5, 45), 100, featureSize)
		dataArray.append(dataPoint)
		# dataPoint2 = prepareFakeData(700 + random.randint(-50, 50), 10 * random.randint(0, 5),
		#                              random.randint(-10, 15), random.uniform(10, 55), 100, featureSize)
		if len(dataArray) % 50 == 0:
			logging.info("Data Generation progress: %i / %i" % (len(dataArray), numberOfLines))
			# print(len(dataArray))

	# data = dataArray.pop()
	# for i in dataArray:  # TODO: Maybe improve performance here? -done
	# 	data = np.append(data, i)
	data = np.concatenate(dataArray)

	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(data['features'], data['labels'], test_size=testSize)

	CsvColumnNames = genColumnNames(featureSize)

	trainFeatureDict = {CsvColumnNames[k]: trainFeatures[:, k] for k in range(2 * featureSize)}
	testFeatureDict = {CsvColumnNames[k]: testFeatures[:, k] for k in range(2 * featureSize)}

	return (trainFeatureDict, trainLabels), (testFeatureDict, testLabels)


def loadFakeDataPandas(featureSize=5, numberOfLines=10, testSize=0.1, numberOfExamples=100):
	dataArray = []

	featureDict = {}
	labelDict = {}

	columns = genColumnNames(featureSize)
	# columns.append('LabelX')
	# columns.append('LabelY')

	for c in columns:
		featureDict[c] = []

	labelDict['LabelX'] = []
	labelDict['LabelY'] = []

	while len(dataArray) < numberOfLines:
		dataPoint = prepareFakeData(random.randint(500, 1200), 10 * random.randint(0, 15),
		                            random.randint(-15, 15), random.uniform(5, 45), numberOfExamples, featureSize)
		dataArray.append(dataPoint)
		# dataPoint2 = prepareFakeData(700 + random.randint(-50, 50), 10 * random.randint(0, 5),
		#                              random.randint(-10, 15), random.uniform(10, 55), 100, featureSize)
		if len(dataArray) % 50 == 0:
			logging.info("Data Generation progress: %i / %i" % (len(dataArray), numberOfLines))
		# print(len(dataArray))

	for dataPoint in dataArray:
		for elem in dataPoint:

			assert len(columns) == len(elem[0])
			assert len(elem[1]) == 2

			for i in range(len(elem[0])):
				featureDict[columns[i]].append(elem[0][i])

			labelDict['LabelX'].append(elem[1][0])
			labelDict['LabelY'].append(elem[1][1])

	featureDf = pd.DataFrame(featureDict)
	labelDf = pd.DataFrame(labelDict)

	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(featureDf, labelDf, test_size=testSize)

	return (trainFeatures, trainLabels), (testFeatures, testLabels)


def prepareRawMeasSeparation(inputFile, featureSize=5, separatorPosY=1550, predictionCutOff=1300, direction=True):
	"""loads real(!) data from a csv file return a dataframe"""
	
	# dt = np.dtype([('features', float, (2 * featureSize,)), ('labels', float, (2,))])
	
	logging.info("Preparing Data from " + inputFile)
	
	df = pd.read_csv(inputFile)
	df = _validateDF(df, featureSize)
	
	numberTracks = (df.shape[1]) / 2
	
	assert numberTracks == int(numberTracks)
	
	numberTracks = int(numberTracks)
	
	columnNames = genColumnNames(featureSize)
	columnNames.append('LabelPosBalken')
	columnNames.append('LabelTime')
	
	data = []
	
	for i in range(numberTracks):
		trackNo = int(''.join([s for s in df.iloc[:,2*i].name if s.isdigit()]))
		
		if direction:
			a = df.iloc[:, (2 * i)].values
			b = df.iloc[:, (2 * i + 1)].values
		else:
			a = df.iloc[:, (2 * i + 1)].values
			b = df.iloc[:, (2 * i)].values
		
		a = _removeNans(a)
		b = _removeNans(b)
		
		if np.isnan(a).any() or np.isnan(b).any():
			logging.warning("skipping track {}: NaN values in track".format(trackNo))
			continue
	
		assert len(a) == len(b)
		
		
		# sort out vanishing tracks
		if max(b) < separatorPosY:
			logging.warning("skipping track {}: highest Value smaller than separator".format(trackNo))
			continue
		
		indexCut = bisect.bisect_left(b, predictionCutOff)
		if indexCut - featureSize < 0:
			logging.warning("skipping track {}: Not enough elements for features".format(trackNo))
			continue
		
		# get position above separator
		xLoc = -1
		if separatorPosY not in b:
			postVal = b[b > separatorPosY].min()
			preVal = b[b < separatorPosY].max()
			preLoc = np.where(b == preVal)
			assert len(preLoc) == len(preLoc[0]) == 1
			assert int(preLoc[0][0]) == preLoc[0][0]
			preLoc = preLoc[0][0]
			
			postLoc = np.where(b == postVal)
			assert len(postLoc) == len(postLoc[0]) == 1
			assert int(postLoc[0][0]) == postLoc[0][0]
			postLoc = postLoc[0][0]
		
			logging.debug("Track {}. positions: preloc {} - postloc {} \n preVal {} - postval {}".format(trackNo, preLoc, postLoc, preVal, postVal))
			
			if preLoc + 1 != postLoc:
				logging.warning("skipping track {}: uncertain Error")
				continue
			
			assert preLoc + 1 == postLoc
			
			L1 = _line([0, separatorPosY], [2000, separatorPosY])
			L2 = _line([a[preLoc], b[preLoc]], [a[postLoc], b[postLoc]])
			
			R = _intersection(L1, L2)
			if R:
				logging.debug("intersection found! {}".format(R))
				xLoc = R[0]
				additionalDistance = _distanceEu((a[preLoc], b[preLoc]), R) / _distanceEu((a[preLoc], b[preLoc]), (a[postLoc], b[postLoc]))
			else:
				logging.error("no intersection found for track {}".format(trackNo))
				sys.exit(1)
		else:
			logging.debug("track {}: value in array!".format(trackNo))
			xLoc = a[np.where(b == separatorPosY)]
			additionalDistance = 0
			
		# remove items up to prediction phase
		

		
		a = a[:indexCut]
		b = b[:indexCut]
		
	
		featureCount = len(a) - featureSize
		
		for k in range(featureCount):
			temp = {}
			temp['features'] = np.append(a[k:(k+ featureSize)], b[k:(k+featureSize)])
			temp['labels'] = np.append(xLoc, abs(preLoc - k) + additionalDistance)
			data.append(temp)

	# newDf = pd.DataFrame(columns=columNames)
	dataFrameList = []

	for elem in data:
		# print(elem)
		tempdict = {columnNames[k]: elem['features'][k] for k in range(2 * featureSize)}
		for i in [-2, -1]:
			tempdict[columnNames[i]] = elem['labels'][i]

		dataFrameList.append(pd.DataFrame(tempdict, index=[0]))


	# assert len(dataFrameList) > 0
	if len(dataFrameList) != 0:
		newDf = pd.concat(dataFrameList, ignore_index=True)

		sizeOld = newDf.shape[0]

		newDf.dropna(subset=[columnNames[-1], columnNames[-2]], inplace=True)

		if sizeOld != newDf.shape[0]:
			logging.info("removed(s) Row for Label NaN")

		sizeOld = newDf.shape[0]
		newDf.dropna(axis=0, inplace=True)

		if sizeOld != newDf.shape[0]:
			logging.info("removed Row(s) for Feature NaN")

	else:
		newDf = pd.DataFrame()
		logging.warning("no data found in " + inputFile)

	return newDf


def loadRawMeasSeparation(input, featureSize=5, testSize=0.1, separatorPosY=1550, predictionCutOff=1300):
	"""loads all the raw measurement data from input into a pandas Dataframe and """
	
	# if input[:5] == "/home":
	# 	folder = input
	# else:
	# 	folder = '/home/hornberger/Projects/MasterarbeitTobias/' + input
	
	folder = input
	
	# TODO: handle single file inputs? os.path.isDir() - make system agnostic
	
	fileList = []
	if folder[-4:] != '.csv':
		fileList = sorted(glob.glob(folder + '/*.csv'))
		logging.info("getting all csv files in {}".format(folder))
	else:
		fileList.append(folder)
		logging.info("loading file {}".format(folder))
	
	assert len(fileList) > 0, "no files found input location " + folder
	dataFrameList = []
	
	for elem in fileList:
		dataFrameList.append(prepareRawMeasSeparation(elem, featureSize, separatorPosY, predictionCutOff))
	
	newDf = pd.concat(dataFrameList, ignore_index=True)
	
	# Remove invalid rows: TODO: more sophisticated approach (forward/backwards fill?)
	
	if not pd.notnull(newDf).all().all():
		raise ValueError("NaNs in labels or feature break Neural Network")
	
	labelDf = newDf[['LabelPosBalken', 'LabelTime']].copy()
	featureDf = newDf.drop(['LabelPosBalken', 'LabelTime'], axis=1)
	
	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(featureDf, labelDf, test_size=testSize)
	
	return (trainFeatures, trainLabels), (testFeatures, testLabels)


def _findSeparationLocation(inputFile, featureSize, separatorPosY):
	
	logging.info("Preparing Data from " + inputFile)
	
	df = pd.read_csv(inputFile)
	df = _validateDF(df, featureSize)
	
	numberTracks = (df.shape[1]) / 2
	
	assert numberTracks == int(numberTracks)
	
	xLast = []
	yLast = []
	xSecondToLast = []
	ySecondToLast = []
	
	data = []
	
	for i in range(numberTracks):
		a = df.iloc[:, (2 * i)].values
		b = df.iloc[:, (2 * i + 1)].values
		
		# a = a[~np.isnan(a)]   #Remove nans from columns
		# b = b[~np.isnan(b)]   #Remove NaNs from Columns
		
		a = _removeNans(a)
		b = _removeNans(b)
		
		assert len(a) == len(b)
		
		xLast.append(a[-1])
		xSecondToLast.append(a[-2])
		
		yLast.append(b[-1])
		ySecondToLast.append(b[-2])
		
	print("yLast: max - {}".format(max(yLast)))
	print("yLast: min - {}".format(min(yLast)))
	
	tracksremoved = sum(i < separatorPosY for i in yLast)
	tracksLostElement = sum(i > separatorPosY for i in ySecondToLast)
	
	print("Tracks removed by Separator Position: {}".format(tracksremoved))
	print("Tracks elements wasted: {}".format(tracksLostElement))
	
	
	return yLast