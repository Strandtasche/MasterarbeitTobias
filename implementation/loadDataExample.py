import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import logging
import glob


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
	# print(df.head(1))

	data = {}
	data["features"] = []
	data["labels"] = []

	numberTracks = (df.shape[1] - 2) / 2

	# print(numberTracks)

	# print(df.iloc[:,2])
	for k in range(50):
		for i in range(int(numberTracks)):
			a = df.iloc[:, (2 + 2 * i)].values
			b = df.iloc[:, (2 + 2 * i + 1)].values
			sp = random.randint(0, df.shape[0] - featureSize - 2)
			final_feature = np.append(a[sp:(sp + featureSize)], b[sp:(sp + featureSize)])
			final_label = np.append(a[sp + featureSize + 1], b[sp + featureSize + 1])

			# print("a: " + str(a[sp + featureSize + 1]) + ", b:" + str(b[sp + featureSize + 1]))

			# print(final_label)
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

	df = dataFrame
	threshold = df.shape[0] - featureSize - 1 # has to fit at least 1 datapoint
	df = df.dropna(axis=1, thresh=threshold)

	# vals_clean = vals[~np.isnan(vals)]

	#TODO: zusammenhängende werte?

	return df


def _removeNans(array):
	arr1d = array
	notnans = np.flatnonzero(~np.isnan(arr1d))
	if notnans.size:
		trimmed = arr1d[notnans[0]: notnans[-1] + 1]  # slice from first not-nan to the last one
	else:
		trimmed = np.zeros(0)

	return trimmed


def prepareRawMeas(inputFile, featureSize=5):
	"""loads real(!) data from a csv file return a dataframe"""

	# dt = np.dtype([('features', float, (2 * featureSize,)), ('labels', float, (2,))])

	logging.info("Preparing Data from " + inputFile)

	df = pd.read_csv(inputFile)
	df = _validateDF(df, featureSize)

	numberTracks = (df.shape[1] - 2) / 2

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
		# print(elem)
		tempdict = {columnNames[k]: elem['features'][k] for k in range(2 * featureSize)}
		for i in [-2, -1]:
			tempdict[columnNames[i]] = elem['labels'][i]

		dataFrameList.append(pd.DataFrame(tempdict, index=[0]))


	# assert len(dataFrameList) > 0
	if len(dataFrameList) != 0:
		newDf = pd.concat(dataFrameList, ignore_index=True)
	else:
		newDf = pd.DataFrame()
		logging.warning("no data found in " + inputFile)

	# print(testFeatures)
	# print(testLabels)

	return newDf



def loadRawMeas(input, featureSize=5, testSize=0.1):

	folder = ''
	if input[:5] == "/home":
		folder = input
	else:
		folder = '/home/hornberger/Projects/MasterarbeitTobias/' + input


	fileList = glob.glob(folder + '/*')

	assert len(fileList) > 0, "no files found input location " + folder + '/'
	dataFrameList = []

	for elem in fileList:
		dataFrameList.append(prepareRawMeas(elem, featureSize))

	newDf = pd.concat(dataFrameList, ignore_index=True)

	labelDf = newDf[['LabelX', 'LabelY']].copy()
	featureDf = newDf.drop(['LabelX', 'LabelY'], axis=1)

	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(featureDf, labelDf, test_size=testSize)

	return (trainFeatures, trainLabels), (testFeatures, testLabels)



def loadData(featureSize=5):
	"""generates data in the right format from csv files and returns a
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
	# for i in dataArray:  # TODO: Maybe improve performance here?
	# 	data = np.append(data, i)
	data = np.concatenate(dataArray)

	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(data['features'], data['labels'], test_size=testSize)

	CsvColumnNames = genColumnNames(featureSize)

	trainFeatureDict = {CsvColumnNames[k]: trainFeatures[:, k] for k in range(2 * featureSize)}
	testFeatureDict = {CsvColumnNames[k]: testFeatures[:, k] for k in range(2 * featureSize)}

	return (trainFeatureDict, trainLabels), (testFeatureDict, testLabels)
