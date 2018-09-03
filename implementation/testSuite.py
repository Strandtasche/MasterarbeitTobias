import MaUtil
import loadDataExample as ld
import os
from numpy import *
import logging
import pandas as pd

from MaUtil import *

import argparse

from tensorflow import estimator
import customEstimator as cE
import tensorflow as tf

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--training', help='if training of eval', action="store_true")
parser.add_argument('--separator', nargs='*', type=int)


def test1():
	print("Running testSuite")
	
	(fakeTrainFeature, fakeTrainLabel), (fakeTestFeature, fakeTestLabel) = ld.loadFakeDataPandas(featureSize=5,
																								 numberOfLines=1)
	
	print("Data loaded")
	
	exampleDF = fakeTrainFeature.head(1)
	exampleLabel = fakeTrainLabel.head(1)
	
	i = 10
	falsePredict = [[array([0 * i, 0 * i], dtype=float32)], [array([5 * i, 5 * i], dtype=float32)],
					[array([10 * i, 10 * i], dtype=float32)], [array([15 * i, 15 * i], dtype=float32)],
					[array([20 * i, 20 * i], dtype=float32)], [array([25 * i, 25 * i], dtype=float32)]]
	
	MaUtil.plotTrainDataPandas(exampleDF, exampleLabel, falsePredict, '/home/hornberger/testSuite')
	
	print("finished")


def test2():
	df = ld.prepareRawMeasNextStep(
		'/home/hornberger/Projects/MasterarbeitTobias/data/experiment01/07_data_trackHistory_NothingDeleted.csv', 5)
	print(df)


def test3(loading=False):
	print("loading data")
	
	if loading == False:
		(X_train1, y_train1), (X_test1, y_test1) = ld.loadRawMeasNextStep(
			'/home/hornberger/Projects/MasterarbeitTobias/data/experiment01', 5, 0.1)
		(X_train2, y_train2), (X_test2, y_test2) = ld.loadFakeDataPandas(5, 500, 0.1)
		with pd.HDFStore('data.h5') as store:
			store['xtrain1'] = X_train1
			store['ytrain1'] = y_train1
			
			store['xtest1'] = X_test1
			store['ytest1'] = y_test1
			
			store['xtrain2'] = X_train2
			store['ytrain2'] = y_train2
			
			store['xtest2'] = X_test2
			store['ytest2'] = y_test2
	else:
		try:
			logging.info("loading data from store")
			
			with pd.HDFStore('data.h5') as store:
				X_train1 = store['xtrain1']
				y_train1 = store['ytrain1']
				
				X_test1 = store['xtest1']
				y_test1 = store['ytest1']
				
				X_train2 = store['xtrain2']
				y_train2 = store['ytrain2']
				
				X_test2 = store['xtest2']
				y_test2 = store['ytest2']
		except:
			# logging.error("Error while loading from pickled data")
			logging.error("Error while loading from stored data")
			exit()
	
	print(X_train1[X_train1.isna().any(axis=1)])  # [X_train1.isnull().any(axis=1)])


def test4():
	(X_train2, y_train2), (X_test2, y_test2) = ld.loadFakeDataPandas(5, 2, 0.1, 15)
	print(X_test2)
	print(y_test2)
	my_feature_columns = []
	columnNames = ld.genColumnNames(5)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))
	regressor = estimator.Estimator(
		model_fn=cE.myCustomEstimator,
		params={
			"feature_columns": my_feature_columns,
			"learning_rate": 0.001,
			"optimizer": tf.train.AdamOptimizer,
			"hidden_units": [20, 20]
		})
	regressor.train(input_fn=lambda: training_input_fn_Slices(X_train2, y_train2, 1000), steps=1000)
	eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test2, y_test2, 1000))
	print("eval: " + str(eval_dict))
	debug_pred = regressor.predict(input_fn=lambda: eval_input_fn(X_test2, labels=None, batch_size=1000))
	# debug_predicted = [p['predictions'] for p in debug_pred]
	for i in debug_pred:
		print(i)
	print("success!")


def MSE(arr1, arr2):
	diff = [arr1[i] - arr2[i] for i in range(len(arr1))]
	squared = [i ** 2 for i in diff]
	return sum(squared) ** 0.5


def validateAugmentation(loadLoc):
	try:
		with pd.HDFStore(loadLoc) as store:
			featuresTrain = store['xtrain']
			labelsTrain = store['ytrain']
			
			X_test = store['xtest']
			y_test = store['ytest']
	
	except Exception as e:
		logging.error("Error while loading from stored data: {}".format(e))
		sys.exit(1)
	
	plt.axis([100, 2250, 0, 1750])
	
	oldDf = pd.concat([featuresTrain, labelsTrain], axis=1, sort=False)
	
	for i, row in oldDf.iterrows():
		cou = int((len(row) - 2) / 2)
		x = row.iloc[0:cou]
		y = row.iloc[cou:2 * cou]
		plt.plot(x, y, 'ro', label='feature', markersize=4)


def validateAugmentation(loadloc, numberV):
	df = ld.prepareRawMeasNextStep(loadloc, 5)
	
	testSet = df.sample(numberV)
	
	# print(testSet.shape[1])
	# print(testSet)
	# print(testSet.drop(testSet.columns[[-2, -1]], axis=1))
	# print(testSet[['LabelX', 'LabelY']])
	
	augmentsF, augmentsL = augmentData(testSet.drop(testSet.columns[[-2, -1]], axis=1), testSet[['LabelX', 'LabelY']], 1160, False)
	
	# testSet.apply(lambda x: pd.Series(mirrorSingleFeature(x, 1160, False)), axis=1, result_type='broadcast')

	# print(augmentsF)
	
	# augmentsF.drop(testSet.index, axis=0, inplace=True)
	# augmentsL.drop(testSet.index, axis=0, inplace=True)
	
	plt.axis([100, 2250, 0, 1750])
	
	for index, row in testSet.iterrows():
		cou = int((len(row) - 2) / 2)
		x = row[0:cou]
		y = row[cou:2 * cou]
		# print("add")
		# print(x)
		# print(y)
		plt.plot(x, y, 'ro', label='feature', markersize=4)
		plt.plot(row[-2], row[-1], 'go', label='target', markersize=4)
	
	for index, row in augmentsF.iterrows():
		cou = int((len(row)) / 2)
		x = row[0:cou]
		y = row[cou:2*cou]
		# print("{} \n {}".format(x, y))
		plt.plot(x, y, 'm+', label='feature', markersize=8)
	#
	for index, row in augmentsL.iterrows():
		plt.plot(row[0], row[1], 'yx', label='target', markersize=4)
	
	plt.show()


def main(argv):
	# args = parser.parse_args(argv[1:])
	
	# print(args.separator)

	loc = '/home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugeln_001_trackHistory_NothingDeleted.csv'
	validateAugmentation(loc, 1)

# if __name__ == '__main__':
# 	tf.logging.set_verbosity(tf.logging.ERROR)
# 	logging.basicConfig(level=logging.INFO)
# 	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 	logging.info('Tensorflow %s' % tf.__version__)
# 	tf.app.run(main)
