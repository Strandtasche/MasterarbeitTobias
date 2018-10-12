import datetime
import logging
import sys
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from adjustText import adjust_text

from evaluationHelper import prepareEvaluationNextStepCVCA, prepareEvaluationSeparatorCVCA, predictionConstantVelSeparator


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
	MODEL_PATH += 'FS%s_' % hyperparams.arch.feature_size
	MODEL_PATH += 'LR%s_' % (str(hyperparams.train.learning_rate % 1)[2:])
	
	if fake:
		MODEL_PATH += 'FD%s' % hyperparams.data.numberFakeLines
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


# kopiert aus Stackoverflow thread "https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python"f
def _line(p1, p2):
	"""helper function, which returns the representation a line through the 2 given points"""
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0] * p2[1] - p2[0] * p1[1])
	return A, B, -C


def _intersection(L1, L2):
	"""helper function, which returns the intersection point between 2 lines defined by _line"""
	D = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0:
		x = Dx / D
		y = Dy / D
		return x, y
	else:
		return False


def _distanceEu(p1, p2):
	"""helper function, which returns the euclidian distance between two points"""
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


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
		
		plt.plot(x, y, 'ro', label='function to predict')
		plt.plot(y_vals2[k][0], y_vals2[k][1], 'gx', label='target')
		plt.plot(y_predicted[k][0], y_predicted[k][1], 'b+', label='prediction')
	plt.plot()
	# plt.legend(loc=9)
	
	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	plt.xlabel('x-Coordinate (px)')
	plt.ylabel('y-Coordinate (px)')
	plt.savefig(MODEL_PATH + '_' + time_stamp + '.png', dpi=300)
	# plt.show()
	plt.close()


def plotDataNextStepPandas(numberPrint, x_pred2, y_vals2, y_predicted, savePath, lim, units, name=None):
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
	plt.xlim(lim[0], lim[1])
	plt.ylim(lim[2], lim[3])
	# plt.xlabel('x-Coordinate (px)')
	# plt.ylabel('y-Coordinate (px)')
	plt.xlabel('x-Coordinate in {}'.format(units['loc']))
	plt.ylabel('y-Coordinate in {}'.format(units['loc'])) # a little bit hacky
	
	plt.title('%s DNNRegressor NextStep' % savePath.split('/')[-1])
	
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), loc=4)
	
	logging.info("Saving Image to file {}".format(output))
	# plt.tight_layout()
	plt.savefig(output, dpi=900)
	# plt.show()
	plt.close()


def plotTrainDataPandas(x_pred2, y_vals2, y_predicted, savePath, units):
	"""plot the different locations of prediction during training"""
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	
	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].values
	x_t = y_vals2[['LabelX']].values
	y_t = y_vals2[['LabelY']].values
	
	# print(x)
	# print(y)
	
	assert len(x) == len(y)
	
	plt.plot(x, y, 'ro', label='function to predict')
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
	plt.xlabel('x-Coordinate in {}'.format(units['loc']))
	plt.ylabel('y-Coordinate in {}'.format(units['loc'])) # a little bit hacky
	logging.info("Saving debug image to file {}".format(savePath + '_' + time_stamp + '.png', ))
	plt.savefig(savePath + '_' + time_stamp + '.png', dpi=300)
	# plt.show()
	plt.close()


def plotDataSeparatorPandas(numberPrint, x_pred2, y_vals2, separatorPosition, y_predicted, savePath, lim, units, name=None):
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
		plt.plot(ausgleichsgeradenX[i], ausgleichsgeradenY[i], color='orange', linestyle='dashed',
				 label='best fit straight line')
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
	plt.xlim(lim[0], lim[1])
	plt.ylim(lim[2], lim[3])
	
	plt.title('%s DNNRegressor Separator' % savePath.split('/')[-1])
	plt.xlabel('x-Coordinate in {}'.format(units['loc']), labelpad=None)
	plt.ylabel('y-Coordinate in {}'.format(units['loc']), labelpad=None) # a little bit hacky
	logging.info("Saving Image to file {}".format(output))
	# plt.tight_layout()
	plt.savefig(output, dpi=900)
	# plt.show()
	plt.close()


def prepareMaximumLossAnalysisNextStep(X_test, y_test, numberPrint, regressor, batchSize):
	"""a helper function for maximum loss analysis in nextStep-Prediction mode,
	returns a pandas dataframe with features, labels, prediction by the net and the MSE of that prediction"""
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
	"""a helper function for maximum loss analysis in separator-Prediction mode,
	returns a pandas dataframe with features, labels, prediction by the net and the MSE of that prediction"""
	totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(X_test, labels=None, batch_size=batchSize))
	totalPredictions = [p['predictions'] for p in totalPredictGen]
	intersectPredL = [p[0] for p in totalPredictions]
	timePredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'PredictionIntersect': intersectPredL, 'PredictionTime': timePredL},
							  index=y_test.index,
							  columns=['PredictionIntersect', 'PredictionTime'])
	pandasLost = pd.concat([X_test, y_test, pandasLost], axis=1)
	pandasLost['positionError'] = pandasLost.apply(
		lambda row: abs(row['LabelPosBalken'] - row['PredictionIntersect']), axis=1)
	maximumLossAnalysisCount = numberPrint  # TODO: potenziell variable mit Arg? / separat?
	printDF = pandasLost.sort_values(by='positionError', ascending=False).head(maximumLossAnalysisCount)
	return printDF


def evaluateResultNextStep(X_test, y_test, totalPredictions):
	"""function to evaluate the result of this nets nextStep prediction. no return value,
	but writes a description of the error distribution and shows some plots for better visualisation"""
	
	xPredL = [p[0] for p in totalPredictions]
	yPredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'NNPredictionX': xPredL, 'NNPredictionY': yPredL}, index=y_test.index)
	
	caCvDfnextStep = prepareEvaluationNextStepCVCA(X_test)
	

	pandasLost = pd.concat([X_test, y_test, pandasLost, caCvDfnextStep], axis=1)
	
	pandasLost['NNpixelErrorX'] = pandasLost.apply(
		lambda row: (row['LabelX'] - row['NNPredictionX']), axis=1)
	pandasLost['NNpixelErrorY'] = pandasLost.apply(
		lambda row: (row['LabelY'] - row['NNPredictionY']), axis=1)
	pandasLost['NNpixelErrorTotal'] = pandasLost.apply(
		lambda row: ((row['LabelX'] - row['NNPredictionX']) ** 2 + (row['LabelY'] - row['NNPredictionY']) ** 2) ** 0.5,
		axis=1)
	
	pandasLost['CVpixelErrorX'] = pandasLost.apply(
		lambda row: (row['LabelX'] - row['CV_Prediction_X']), axis=1)
	pandasLost['CVpixelErrorY'] = pandasLost.apply(
		lambda row: (row['LabelY'] - row['CV_Prediction_Y']), axis=1)
	pandasLost['CVpixelErrorTotal'] = pandasLost.apply(
		lambda row: ((row['LabelX'] - row['CV_Prediction_X']) ** 2 + (row['LabelY'] - row['CV_Prediction_Y']) ** 2) ** 0.5,
		axis=1)
	pandasLost['CApixelErrorX'] = pandasLost.apply(
		lambda row: (row['LabelX'] - row['CA_Prediction_X']), axis=1)
	pandasLost['CApixelErrorY'] = pandasLost.apply(
		lambda row: (row['LabelY'] - row['CA_Prediction_Y']), axis=1)
	pandasLost['CApixelErrorTotal'] = pandasLost.apply(
		lambda row: ((row['LabelX'] - row['CA_Prediction_X']) ** 2 + (row['LabelY'] - row['CA_Prediction_Y']) ** 2) ** 0.5,
		axis=1)
	
	relevantColumns = ['NNpixelErrorX', 'NNpixelErrorY', 'NNpixelErrorTotal',
							 'CVpixelErrorX', 'CVpixelErrorY', 'CVpixelErrorTotal',
							 'CApixelErrorX', 'CApixelErrorY', 'CApixelErrorTotal']
	
	reducedRelColumns = [relevantColumns[2], relevantColumns[5], relevantColumns[8]]
	
	_printPDfull(pandasLost[reducedRelColumns].describe())

	logging.info("number of predictions with error > 3: {}".format((pandasLost['NNpixelErrorTotal'] > 3).sum()))
	
	# TODO: Maybe save column total Pixelerror of current prediction so it can be compared to other schüttgüter
	
	fig1, ax1 = plt.subplots()
	ax1.boxplot([pandasLost['NNpixelErrorTotal'], pandasLost['CVpixelErrorTotal'], pandasLost['CApixelErrorTotal']], showfliers=False)
	# plt.show()
	fig2, ax2 = plt.subplots()
	ax2.hist([pandasLost['NNpixelErrorTotal'], pandasLost['CVpixelErrorTotal'], pandasLost['CApixelErrorTotal']],
			 bins=40, label=['NNpixelErrorTotal', 'CVpixelErrorTotal', 'CApixelErrorTotal'])
	# # plt.yscale('log')
	# ax.style.use('seaborn-muted')
	plt.legend(loc=1)
	plt.show()


def _printPDfull(dataframe):
	pd.set_option('display.max_colwidth', -1)
	pd.set_option('display.max_columns', None)
	logging.info("\n{}".format(dataframe))
	pd.reset_option('display.max_columns')
	pd.reset_option('display.max_colwidth')


def _getRelIndices(columns, separator, direction):
	if not separator:
		if direction:
			relIndices = [i for i in range(int((len(columns) - 2) / 2))]
			relIndices.append(len(columns) - 2)
		else:
			relIndices = [i for i in range(int((len(columns) - 2) / 2), len(columns) - 2)]
			relIndices.append(len(columns) - 1)
	else:
		if direction:
			relIndices = [i for i in range(int((len(columns) - 2) / 2))]
			relIndices.append(len(columns) - 2)
		else:
			relIndices = [i for i in range(int((len(columns) - 2) / 2), len(columns) - 2)]
			relIndices.append(len(columns) - 2)
	return relIndices


def mirrorSingleFeature(points, midpoint, relIndices):
	"""helper function for Data Augmentation instance - mirror input
	given an array of points and a line to mirror them on, returns an array of mirrored coordinates"""
	
	newpoints = points
	
	
	# print(relIndices)
	for i in relIndices:
		if points[i] < midpoint:
			newpoints[i] = points[i] + 2 * abs(midpoint - points[i])
		else:
			newpoints[i] = points[i] - 2 * abs(midpoint - points[i])
	
	return newpoints


def augmentData(featuresTrain, labelsTrain, midpoint, augmentRange, separator, direction=True):
	"""applies the mirror input data augmentation to some data"""
	
	oldDf = pd.concat([featuresTrain, labelsTrain], axis=1, sort=False)
	newDf = oldDf.copy()
	newDf.index += oldDf.index.max()
	
	relIndices = _getRelIndices(newDf.columns, separator, direction)
	# sizeBefore = newDf.shape
	newDf = newDf[(newDf.iloc[:,relIndices] > (midpoint - augmentRange)).any(axis=1) & (newDf.iloc[:,relIndices] < (midpoint + augmentRange)).any(axis=1)]
	# sizeAfter = newDf.shape
	# print("before: {}, after: {}".format(sizeBefore, sizeAfter))
	# newDf.apply(lambda x: pd.Series(mirrorSingleFeature(x, midpoint, separator, direction)), axis=1, result_type='broadcast')
	newDf.apply(lambda x: pd.Series(mirrorSingleFeature(x, midpoint, relIndices)), axis=1,
				result_type='broadcast')
	
	assert pd.DataFrame.equals(oldDf, newDf) == False
	
	newDf = pd.concat([oldDf, newDf])
	
	if separator:
		augmentedLabelDf = newDf[['LabelPosBalken', 'LabelTime']].copy()
		augmentedFeatureDf = newDf.drop(['LabelPosBalken', 'LabelTime'], axis=1)
	else:
		augmentedLabelDf = newDf[['LabelX', 'LabelY']].copy()
		augmentedFeatureDf = newDf.drop(['LabelX', 'LabelY'], axis=1)
	
	return augmentedFeatureDf, augmentedLabelDf


def getMedianAccel(X_test, separator, direction):
	if not direction: # moving along x axis
		relCols = X_test.columns[0:int(len(X_test.columns)/2)]
		# print("x axis")
	else: # moving along y axis
		relCols = X_test.columns[int(len(X_test.columns)/2):len(X_test.columns)]
		# print("y axis")

	accel = (X_test.loc[:, relCols[-1]] - X_test.loc[:, relCols[-2]]) - (X_test.loc[:, relCols[-2]] - X_test.loc[:, relCols[-3]])
	
	return accel.median()
	# X_test['accel'] = X_test.apply(
	# 	lambda row: (row[relCols[-1]] - row[relCols[-2]]) - (row[relCols[-2]] - row[relCols[-3]]), axis=1
	# )
	#
	# return X_test['accel'].median()


def getOptimalAccel(X_test, y_test, separatorPosition, direction):
	logging.info("getting optimal accel for {} examples".format(X_test.shape[0]))
	if not direction: # moving along x axis
		relCols = X_test.columns[0:int(len(X_test.columns)/2)]
	# print("x axis")
	else: # moving along y axis
		relCols = X_test.columns[int(len(X_test.columns)/2):len(X_test.columns)]
	# print("y axis")
	tempDf = pd.concat([X_test, y_test], axis=1)
	
	tempDf['optAc'] = tempDf.apply(
		lambda row: (separatorPosition - row[relCols[-1]] - row['LabelTime'] * (row[relCols[-1]] - row[relCols[-2]]))/(0.5 * row['LabelTime'] ** 2)
		, axis=1)
	
	return tempDf['optAc'].median()


def filterDataForIntersection(dataSet, thresholdPoint, direction):
	if direction:
		columnNameLast = dataSet.columns[-1]
		columnNamePenultimate = dataSet.columns[-2]
	else:
		columnNameLast = dataSet.columns[int(len(dataSet.columns)/2)]
		columnNamePenultimate = dataSet.columns[int(len(dataSet.columns)/2)-1]
	dataSet = dataSet[(dataSet[columnNameLast] > thresholdPoint) & (dataSet[columnNamePenultimate] < thresholdPoint)]
	
	return dataSet

def getCVBias(dataSetFeatures, dataSetLabels, separatorPosition, direction):
	
	# locPredCV = []
	timePredCV = []
	for index, row in dataSetFeatures.iterrows():
		_, timeSepCV = predictionConstantVelSeparator(row.values, separatorPosition, direction)
		# locPredCV.append(locSepCV)
		timePredCV.append(timeSepCV)
		
	dataInp = {'CV_Prediction_Time': timePredCV}
	constantVel = pd.DataFrame(data=dataInp, index=dataSetFeatures.index)
	
	timeError = dataSetLabels['LabelTime'] - constantVel['CV_Prediction_Time'] * 100
	
	return timeError.median()
	


def evaluateResultSeparator(X_test, y_test, totalPredictions, separatorPosition, thresholdPoint, configDict, units, direction):
	"""function to evaluate the result of this nets nextStep prediction.
	no return value, but writes a description of the error distribution and shows some plots for better visualisation"""
	
	xPredL = [p[0] for p in totalPredictions]
	timePredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'NNPredictionPosBalken': xPredL, 'NNPredictionTime': timePredL}, index=y_test.index)

	# medianAccel = configDict['medAc']
	# optimalAccel = configDict['optAc']
	# cvBias = configDict['cvBias']
	# logging.info("cvBias: {}".format(cvBias))
	# thresholdPoint = predictionCutOff - 100  # TODO: SUPER TEMPORARY, DO NOT USE IN FINAL SITUATION!
	
	filteredFeatures = filterDataForIntersection(X_test, thresholdPoint, direction)
	
	comboAlign = pd.concat([filteredFeatures, y_test.loc[filteredFeatures.index]], axis=1)
	
	logging.info("Evaluation on {} data points".format(comboAlign.shape[0]))
	
	tempDf = comboAlign.sample(frac=0.05)
	#
	# assert tempDf.shape[0] != 0
	#
	# medianAccel = _getMedianAccel(tempDf[X_test.columns], True, direction)
	# altOptimalAccel = getOptimalAccel(tempDf[X_test.columns], tempDf[y_test.columns].loc[tempDf.index], separatorPosition, direction)
	caCvDfSeparator = prepareEvaluationSeparatorCVCA(comboAlign[X_test.columns], configDict, separatorPosition, direction)
	
	pandasLost = pd.concat([comboAlign, pandasLost, caCvDfSeparator], axis=1)
	pandasLost.dropna(inplace=True)
	
	pandasLost['NNpixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['NNPredictionPosBalken']), axis=1)
	pandasLost['NNerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - row['NNPredictionTime']), axis=1)
	
	pandasLost['CVpixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['CV_Prediction_Loc']), axis=1) # TODO: Handle Scaling!
	pandasLost['CVerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - row['CV_Prediction_Time'] * 100), axis=1) # factor 100 because of scaling
	pandasLost['CVBCpixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['CVBC_Prediction_Loc']), axis=1) # TODO: Handle Scaling!
	pandasLost['CVBCerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - row['CVBC_Prediction_Time'] * 100), axis=1) # factor 100 because of scaling
	
	pandasLost['CApixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['CA_Prediction_Loc']), axis=1)
	pandasLost['CAerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - row['CA_Prediction_Time'] * 100), axis=1) # factor 100 because of scaling
	
	pandasLost['AApixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['AA_Prediction_Loc']), axis=1)
	pandasLost['AAerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - (row['AA_Prediction_Time'] * 100)), axis=1) # factor 100 because of scaling
	
	pandasLost['IApixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['IA_Prediction_Loc']), axis=1)
	pandasLost['IAerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - (row['IA_Prediction_Time'] * 100)), axis=1) # factor 100 because of scaling
	
	
	# TODO: optional: change adding of new column from pandas apply to...
	# something like '	accel = (X_test.iloc[:,relCols[-1]] - X_test.iloc[:,relCols[-2]])'
	
	relevantColumnsLoc = ['NNpixelErrorPosBalken', 'CVpixelErrorPosBalken', 'CVBCpixelErrorPosBalken', 'CApixelErrorPosBalken']
	relevantColumnsTime = ['NNerrorTime', 'CVerrorTime', 'CVBCerrorTime', 'CAerrorTime', 'AAerrorTime', 'IAerrorTime']
	# logging.info("\n{}".format(pandasLost[relevantColumns]))
	
	_printPDfull(pandasLost[relevantColumnsLoc].describe())
	_printPDfull(pandasLost[relevantColumnsTime].describe())

	# logging.info("number of predictions with error > 3: {}".format((pandasLost['NNpixelErrorTotal'] > 3).sum()))
	plt.rc('grid', linestyle=":")
	fig1, ax1 = plt.subplots()
	# ax1.grid(True)
	ax1.yaxis.grid(True)
	ax1.boxplot([pandasLost[i] for i in relevantColumnsLoc], showfliers=False, labels=relevantColumnsLoc)
	xtickNames = plt.setp(ax1, xticklabels=relevantColumnsLoc)
	plt.setp(xtickNames, rotation=45, fontsize=8)
	ax1.set_title('Boxplot Location Error')
	ax1.set_ylabel('error in {}'.format(units['loc']))  # a little bit hacky
	fig1.tight_layout()

	# plt.show()
	fig2, ax2 = plt.subplots()
	ax2.hist([pandasLost[i] for i in relevantColumnsLoc],
			 bins=40, label=relevantColumnsLoc)
	ax2.set_title('Histogram Location Error')
	ax2.set_ylabel('# of elements in bin')  # a little bit hacky
	ax2.legend(loc=1)
	# # plt.yscale('log')
	# ax.style.use('seaborn-muted')
	
	fig3, ax3 = plt.subplots()
	# ax3.grid(True)
	ax3.yaxis.grid(True)
	ax3.boxplot([pandasLost[i] for i in relevantColumnsTime], showfliers=False)
	xtickNamesAx3 = plt.setp(ax3, xticklabels=relevantColumnsTime)
	plt.setp(xtickNamesAx3, rotation=45, fontsize=8)
	ax3.set_title('Boxplot Time Error')
	ax3.set_ylabel('error in {}'.format(units['time']))  # a little bit hacky
	fig3.tight_layout()
	
	fig4, ax4 = plt.subplots()
	ax4.hist([pandasLost[i] for i in relevantColumnsTime],
			 bins=40, label=relevantColumnsTime)
	ax4.set_title('Histogram Time Error')
	ax4.set_ylabel('# of elements in bin')  # a little bit hacky
	ax4.legend(loc=1)
	plt.tight_layout()
	plt.show()
