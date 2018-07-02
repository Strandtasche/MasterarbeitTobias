import MaUtil
import loadDataExample as ld
import os
from numpy import *
import logging

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

def test3():
	print("loading data")
	(X_train1, y_train1), (X_test1, y_test1) = ld.loadRawMeas('/home/hornberger/Projects/MasterarbeitTobias/data/experiment04', 5, 0.1)
	(X_train2, y_train2), (X_test2, y_test2) = ld.loadFakeDataPandas(5, 500, 0.1)

	print(type(X_test1))



logging.getLogger().setLevel(logging.INFO)

test3()