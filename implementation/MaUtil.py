import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time
import logging
from adjustText import adjust_text
import numpy as np
import pandas as pd
from collections import OrderedDict
import sys


def genModelPath(hyperparams, fake, usingCustomestimator, separator):
	"""returns path to location of model based on parameters"""
	if not fake:
		MODEL_PATH = hyperparams.problem.modelBasePath
	else:
		MODEL_PATH = hyperparams.problem.modelBasePath[0:-1] + "FakeData/"

	if usingCustomestimator:
		MODEL_PATH += 'CustomEstimator_'
	else:
		MODEL_PATH += 'PremadeEstimator_'
	
	if separator:
		MODEL_PATH = MODEL_PATH + 'Separation_'
	else:
		MODEL_PATH = MODEL_PATH + 'PredictNext_'

	for hl in hyperparams.arch.hidden_layers:
		MODEL_PATH += '%s_' % hl
	MODEL_PATH += 'D0%s_' % (int(hyperparams.arch.dropout_rate * 100))
	MODEL_PATH += 'FS%s_' % (hyperparams.arch.feature_size)
	MODEL_PATH += 'LR%s_' % (str(hyperparams.train.learning_rate % 1)[2:])

	if fake:
		MODEL_PATH += 'FD%s' % (hyperparams.data.numberFakeLines)
	else:
		MODEL_PATH += 'DS_%s' % (hyperparams.problem.data_path.replace("/", "_"))

	return MODEL_PATH


def training_input_fn_Slices(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	# featureDict = {ld.CSV_COLUMN_NAMES[k]: features[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))}
	featureDict = dict(features)

	dataset = tf.data.Dataset.from_tensor_slices((featureDict, labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(2500).repeat().batch(batch_size)

	# Return the dataset.

	return dataset


def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	# featureDict = {ld.CSV_COLUMN_NAMES[k]: features[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))}
	featureDict = dict(features)
	if labels is None:
		# No labels, use only features.
		inputs = featureDict
	else:
		inputs = (featureDict, labels)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	# Batch the examples
	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	# Return the dataset.
	return dataset


def plotDataNumpy(numberPrint, x_pred2, y_vals2, y_predicted, savePath):
	"""DEPRECATED!
	plot a certain number of examples (from numpy array) with features, labels and prection and save to .png file"""

	MODEL_PATH = savePath
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	for k in range(numberPrint):
		x = []
		y = []
		for i in x_pred2.keys():
			if i[0] == 'X':
				x.append(x_pred2[i][k])
			else:
				y.append(x_pred2[i][k])

		assert len(x) == len(y)

		plt.plot(x, y, 'ro',label='function to predict')
		plt.plot(y_vals2[k][0], y_vals2[k][1], 'gx', label='target')
		plt.plot(y_predicted[k][0], y_predicted[k][1], 'b+', label='prediction')
	plt.plot()
	#plt.legend(loc=9)

	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	plt.savefig(MODEL_PATH + '_' + time_stamp + '.png', dpi=300)
	# plt.show()
	plt.close()


def plotDataNextStepPandas(numberPrint, x_pred2, y_vals2, y_predicted, savePath, name=None):
	"""plot a certain number of next step prediction examples (from pandas dataframe)
	with features, labels and prection and save to .png file"""
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	if name is None:
		output = savePath + 'Pic_' + time_stamp + '.png'
	else:
		output = name

	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].head(numberPrint).values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].head(numberPrint).values
	x_t = y_vals2[['LabelX']].head(numberPrint).values
	y_t = y_vals2[['LabelY']].head(numberPrint).values

	# print(x)
	# print(y)

	assert len(x) == len(y)

	plt.plot(x, y, 'ro', label='feature', markersize=4)
	plt.plot(x_t, y_t, 'gx', label='label')
	if isinstance(y_predicted, list):
		for k in range(numberPrint):
			plt.plot(y_predicted[k][0], y_predicted[k][1], 'b+', label='prediction')
	else:
		for k in range(numberPrint):
			plt.plot(y_predicted['PredictionX'], y_predicted['PredictionY'], 'b+', label='prediction')
	plt.plot()
	plt.xlim(100, 2250)
	plt.ylim(0, 1750)

	plt.title('%s DNNRegressor NextStep' % savePath.split('/')[-1])
	plt.tight_layout()
	
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), loc=4)

	logging.info("Saving Image to file {}".format(output))
	plt.savefig(output, dpi=900)
	# plt.show()
	plt.close()


def plotTrainDataPandas(x_pred2, y_vals2, y_predicted, savePath):
	"""plot the different locations of prediction during training"""
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].values
	x_t = y_vals2[['LabelX']].values
	y_t = y_vals2[['LabelY']].values

	# print(x)
	# print(y)

	assert len(x) == len(y)

	plt.plot(x, y, 'ro',label='function to predict')
	plt.plot(x_t, y_t, 'gx', label='target')
	i = 0
	total = len(y_predicted)
	textArray = []
	for elem in y_predicted:
		plt.plot(elem[0][0], elem[0][1], 'b+', label='prediction')
		textLabel = "{:.2f}%".format(i / total * 100)
		textArray.append(plt.text(elem[0][0], elem[0][1], textLabel, ha='center', va='center'))
		i = i + 1
	adjust_text(textArray, arrowprops=dict(arrowstyle='->', color='black'))

	plt.plot()
	plt.title('%s DNNRegressor' % savePath.split('/')[-1])
	plt.tight_layout()
	logging.info("Saving debug image to file {}".format(savePath + '_' + time_stamp + '.png',))
	plt.savefig(savePath + '_' + time_stamp + '.png', dpi=300)
	# plt.show()
	plt.close()


def plotDataSeparatorPandas(numberPrint, x_pred2, y_vals2, separatorPosition, y_predicted, savePath, name=None):
	"""plot a certain number of next step prediction examples (from pandas dataframe)
		with features, labels and prection and save to .png file"""
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	
	if name is None:
		output = savePath + 'Pic_' + time_stamp + '.png'
	else:
		output = name
	
	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].head(numberPrint).values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].head(numberPrint).values
	x_t = y_vals2.head(numberPrint).values
	y_t = [separatorPosition] * numberPrint

	ausgleichsgeradenY = []
	ausgleichsgeradenX = []
	for i in range(numberPrint):
		p = np.poly1d(np.polyfit(y[i], x[i], 1))
		p_y = [y[i][0], separatorPosition]
		p_x = [p(y[i][0]), p(separatorPosition)]
		ausgleichsgeradenY.append(p_y)
		ausgleichsgeradenX.append(p_x)
	
	assert len(x) == len(y)
	
	for i in range(len(ausgleichsgeradenY)):
		plt.plot(ausgleichsgeradenX[i], ausgleichsgeradenY[i], color='orange', linestyle='dashed', label='best fit straight line')
	# plt.plot(ausgleichsgeradenX, ausgleichsgeradenY, label='best fit straight line')
	plt.plot(x, y, 'ro', label='feature', markersize=4)
	plt.plot(x_t, y_t, 'gx', label='label')
	if isinstance(y_predicted, list):
		for k in range(numberPrint):
			plt.plot(y_predicted[k][0], separatorPosition, 'b+', label='prediction')
	else:
		if 'PredictionX' in y_predicted:
			for k in range(numberPrint):
				plt.plot(y_predicted['PredictionX'], y_predicted['PredictionY'], 'b+', label='prediction')
		elif 'PredictionIntersect' in y_predicted:
			for k in range(numberPrint):
				plt.plot(y_predicted['PredictionIntersect'].iloc[k], separatorPosition, 'b+', label='prediction')
		else:
			logging.error("weird error in plotting. terminating.")
			sys.exit(-1)
			
	plt.plot()
	plt.xlim(100, 2250)
	plt.ylim(0, 1750)
	
	plt.title('%s DNNRegressor Separator' % savePath.split('/')[-1])
	plt.tight_layout()
	logging.info("Saving Image to file {}".format(output))
	plt.savefig(output, dpi=900)
	plt.show()
	#plt.close()


def prepareMaximumLossAnalysisNextStep(X_test, y_test, numberPrint, regressor, batchSize):
	totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(X_test, labels=None, batch_size=batchSize))
	totalPredictions = [p['predictions'] for p in totalPredictGen]
	xPredL = [p[0] for p in totalPredictions]
	yPredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'PredictionX': xPredL, 'PredictionY': yPredL}, index=y_test.index,
							  columns=['PredictionX', 'PredictionY'])
	pandasLost = pd.concat([X_test, y_test, pandasLost], axis=1)
	pandasLost['MSE'] = pandasLost.apply(
		lambda row: ((row['LabelX'] - row['PredictionX']) ** 2 + (row['LabelY'] - row['PredictionY']) ** 2) / 2, axis=1)
	maximumLossAnalysisCount = numberPrint  # TODO: potenziell variable mit Arg? / separat?
	printDF = pandasLost.sort_values(by='MSE', ascending=False).head(maximumLossAnalysisCount)
	return printDF


def prepareMaximumLossAnalysisSeparator(X_test, y_test, numberPrint, regressor, batchSize):
	totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(X_test, labels=None, batch_size=batchSize))
	totalPredictions = [p['predictions'] for p in totalPredictGen]
	intersectPredL = [p[0] for p in totalPredictions]
	timePredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'PredictionIntersect': intersectPredL, 'PredictionTime': timePredL}, index=y_test.index,
							  columns=['PredictionIntersect', 'PredictionTime'])
	pandasLost = pd.concat([X_test, y_test, pandasLost], axis=1)
	pandasLost['positionError'] = pandasLost.apply(
		lambda row: abs(row['LabelPosBalken'] - row['PredictionIntersect']), axis=1)
	maximumLossAnalysisCount = numberPrint  # TODO: potenziell variable mit Arg? / separat?
	printDF = pandasLost.sort_values(by='positionError', ascending=False).head(maximumLossAnalysisCount)
	return printDF


def evaluateResultNextStep(X_test, y_test, numberPrint, regressor, batchSize):
	totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(X_test, labels=None, batch_size=batchSize))
	totalPredictions = [p['predictions'] for p in totalPredictGen]
	xPredL = [p[0] for p in totalPredictions]
	yPredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'PredictionX': xPredL, 'PredictionY': yPredL}, index=y_test.index,
							  columns=['PredictionX', 'PredictionY'])
	pandasLost = pd.concat([X_test, y_test, pandasLost], axis=1)
	
	pandasLost['pixelErrorX'] = pandasLost.apply(
		lambda row: (row['LabelX'] - row['PredictionX']), axis=1)
	pandasLost['pixelErrorY'] = pandasLost.apply(
		lambda row: (row['LabelY'] - row['PredictionY']), axis=1)
	pandasLost['pixelErrorTotal'] = pandasLost.apply(
		lambda row: ((row['LabelX'] - row['PredictionX']) ** 2 + (row['LabelY'] - row['PredictionY']) ** 2) ** 0.5, axis=1)
	
	print("pixelErrorX.mean: {}".format(pandasLost['pixelErrorX'].mean()))
	print("pixelErrorY.mean: {}".format(pandasLost['pixelErrorY'].mean()))
	print("pixelErrorTotal.mean: {}".format(pandasLost['pixelErrorTotal'].mean()))
	print(pandasLost[['pixelErrorX', 'pixelErrorY', 'pixelErrorTotal']].describe())
	plt.boxplot(pandasLost['pixelErrorX'], showfliers=False)
	#plt.show()