import MaUtil
import loadDataExample as ld
import os
from numpy import *
import logging
import pandas as pd


def test1():
	print("Running testSuite")

	(fakeTrainFeature, fakeTrainLabel), (fakeTestFeature, fakeTestLabel) = ld.loadFakeDataPandas(featureSize=5,
																								 numberOfLines=1)

	print("Data loaded")

	exampleDF = fakeTrainFeature.head(1)
	exampleLabel = fakeTrainLabel.head(1)

	i = 10
	falsePredict = [[array([0*i, 0*i], dtype=float32)], [array([5*i, 5*i], dtype=float32)],
					[array([10*i, 10*i], dtype=float32)], [array([15*i, 15*i], dtype=float32)],
					[array([20*i, 20*i], dtype=float32)], [array([25*i, 25*i], dtype=float32)]]

	MaUtil.plotTrainDataPandas(exampleDF, exampleLabel, falsePredict, '/home/hornberger/testSuite')




	print("finished")

def test2():
	df = ld.prepareRawMeas('/home/hornberger/Projects/MasterarbeitTobias/data/experiment01/07_data_trackHistory_NothingDeleted.csv', 5)
	print(df)

def test3(loading=False):
	print("loading data")

	if loading == False:
		(X_train1, y_train1), (X_test1, y_test1) = ld.loadRawMeas('/home/hornberger/Projects/MasterarbeitTobias/data/experiment01', 5, 0.1)
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

	print(X_train1[X_train1.isna().any(axis=1)]) #[X_train1.isnull().any(axis=1)])

def test4():
	(X_train2, y_train2), (X_test2, y_test2) = ld.loadFakeDataPandas(5, 2, 0.1, 15)

	print(X_test2.shape)

	for index, row in X_test2.iterrows():
		print(X_test2.loc[index])
		print(y_test2.loc[index])



logging.getLogger().setLevel(logging.INFO)

# test3(False)
test4()