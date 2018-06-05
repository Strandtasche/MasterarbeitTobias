import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import random







load = False
# featureSize = 5

# CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5']

CSV_COLUMN_NAMES = ['X_1', 'X_2', 'X_3', 'X_4']

def loadCSVData(file='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv', featureSize=5):

	df = pd.read_csv(file)
	featureSize = 2    # print(df.head(1))


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
			#print(sp)
			final_feature = np.append(a[sp:(sp+featureSize)], b[sp:(sp+featureSize)])
			final_label = np.array(a[sp + featureSize + 1], b[sp + featureSize + 1])

			#print(final_label)
			# assert final_feature.shape[0] == final_label.shape[0]

			data["features"].append(final_feature)
			data["labels"].append(final_label)

	return data


def loadCSVtoNpy(inputFile='/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv', outputfile ='in.npy', featureSize=5):


	dt = np.dtype([('features', float, (2*featureSize,)), ('labels', float )])#(2,))])


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
			x[counter]['labels'] =  a[sp + featureSize + 1] #np.array(a[sp + featureSize + 1], b[sp + featureSize + 1])
			counter = counter + 1

	#print(x)

	np.save(outputfile, x)


def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	print(features)
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	print("this is dataset type: " + str(type(dataset)))
	return dataset

def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	features=dict(features)
	if labels is None:
		# No labels, use only features.
		inputs = features
	else:
		inputs = (features, labels)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	# Batch the examples
	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	# Return the dataset.
	return dataset




def loadData(featureSize=5):
	training_file = '/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case1_labels.csv'
	test_file = '/home/tobi/Projects/KIT/MasterarbeitTobias/data/GroundTruthExample/test_case2_labels_NoNan.csv'

	loadCSVtoNpy(training_file, "training_in.npy", 2)
	loadCSVtoNpy(test_file, "test_in.npy", 2)

	train_data = np.load('training_in.npy')
	train_features = train_data["features"]
	train_labels = train_data["labels"]

	test_data = np.load('test_in.npy')
	test_features = test_data["features"]
	test_labels = test_data["labels"]

	assert train_features.shape[0] == train_labels.shape[0]
	assert test_features.shape[0] == test_labels.shape[0]

	return (train_features, train_labels), (test_features, test_labels)









# print(training_data)
