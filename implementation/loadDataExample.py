import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split






load = False
# featureSize = 5

#CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5']

#CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4']

def genColumnNames(featureSize):
	"""returns an array that can be used as the name for feature columns"""
	names = []
	for i in range(featureSize):
		names.append('X_' + str(i))
	for i in range(featureSize):
		names.append('Y_' + str(i))

	return names


def loadCSVData(file='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv', featureSize=5):
	"""loads data from from CSV file and returns a dictionary with features and labels"""
	df = pd.read_csv(file)
	# print(df.head(1))


	data = {}
	data["features"] = []
	data["labels"] = []

	numberTracks = (df.shape[1] - 2)/2

	# print(numberTracks)

	#print(df.iloc[:,2])
	for k in range(50):
		for i in range(int(numberTracks)):
			a = df.iloc[:,(2 + 2*i)].values
			b = df.iloc[:, (2 + 2*i+ 1)].values
			sp = random.randint(0, df.shape[0] - featureSize - 2)
			final_feature = np.append(a[sp:(sp+featureSize)], b[sp:(sp+featureSize)])
			final_label = np.append(a[sp + featureSize + 1], b[sp + featureSize + 1])

			# print("a: " + str(a[sp + featureSize + 1]) + ", b:" + str(b[sp + featureSize + 1]))

			# print(final_label)
			# assert final_feature.shape[0] == final_label.shape[0]

			data["features"].append(final_feature)
			data["labels"].append(final_label)

	return data


def loadCSVtoNpy(inputFile='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv', outputfile ='in.npy', featureSize=5):
	"""loads data from a csv file and generates a set of features and labels from it.
	the result is saved into a npy file and placed in the current folder"""

	dt = np.dtype([('features', float, (2*featureSize,)), ('labels', float,(2,))])


	df = pd.read_csv(inputFile)
	numberTracks = (df.shape[1] - 2)/2

	assert numberTracks == int(numberTracks)

	numberTracks = int(numberTracks)

	sampleNumberPerTrack = 5

	featureCount = numberTracks * sampleNumberPerTrack

	x = np.zeros((featureCount,), dtype=dt)

	counter = 0

	for i in range(numberTracks):
		a = df.iloc[:,(2 + 2*i)].values
		b = df.iloc[:, (2 + 2*i+ 1)].values
		for k in range(sampleNumberPerTrack):
			sp = random.randint(0, df.shape[0] - featureSize - 2)
			x[counter]['features'] = np.append(a[sp:(sp+featureSize)], b[sp:(sp+featureSize)])
			x[counter]['labels'] = np.append(a[sp + featureSize + 1], b[sp + featureSize + 1]) #a[sp + featureSize + 1]
			counter = counter + 1


	np.save(outputfile, x)




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

	trainFeatureDict = {CsvColumnNames[k]: train_features[:,k] for k in range(2*featureSize)}
	testFeatureDict = {CsvColumnNames[k]: test_features[:,k] for k in range(2*featureSize)}

	# labelNames = ['X_n', 'Y_n']


	# trainLabelDict = {labelNames[k]: train_labels[:,k] for k in range(2)}
	# testLabelDict = {labelNames[k]: test_labels[:,k] for k in range(2)}


	assert train_features.shape[0] == train_labels.shape[0]
	assert test_features.shape[0] == test_labels.shape[0]

	return (trainFeatureDict, train_labels), (testFeatureDict, test_labels)


def prepareFakeData(startpX = 700, startpY = 0, veloX=-5, veloY=30, numberEx = 75, featureSize=5):
	startpointx = startpX
	startpointy = startpY

	numberExamples = numberEx

	speedx = veloX
	speedy = veloY

	posX = [startpointx + i*speedx for i in range(numberExamples)]
	posY = [startpointy + i*speedy for i in range(numberExamples)]

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
		data[i]['features'] = np.append(posX[i:(i+featureSize)], posY[i:(i+featureSize)])
		data[i]['labels'] = np.append(posX[i + featureSize], posY[i + featureSize])

	return data


def loadFakeData(featureSize=5, numberOfLines=20):

	dataArray = []

	# for i in range(numberOfLines):
	while len(dataArray) < numberOfLines:

		dataPoint = prepareFakeData(random.randint(500,1200), 10 * random.randint(0, 15),
		random.randint(-15, 15), random.uniform(5, 45), 100, featureSize)
		dataArray.append(dataPoint)

	data = dataArray.pop()
	for i in dataArray:    #TODO: Maybe improve performance here?
		data = np.append(data, i)


	trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(data['features'], data['labels'], test_size=0.1)

	CsvColumnNames = genColumnNames(featureSize)
	
	trainFeatureDict={CsvColumnNames[k]: trainFeatures[:, k] for k in range(2 * featureSize)}
	testFeatureDict = {CsvColumnNames[k]: testFeatures[:, k] for k in range(2 * featureSize)}

	return (trainFeatureDict, trainLabels), (testFeatureDict, testLabels)


