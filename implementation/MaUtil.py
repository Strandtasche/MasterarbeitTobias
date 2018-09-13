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
	plt.xlabel('x-Coordinate (px)')
	plt.ylabel('y-Coordinate (px)')
	
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
	plt.xlabel('x-Coordinate (px)')
	plt.ylabel('y-Coordinate (px)')
	logging.info("Saving debug image to file {}".format(savePath + '_' + time_stamp + '.png', ))
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
	plt.xlim(100, 2250)
	plt.ylim(0, 1750)
	
	plt.title('%s DNNRegressor Separator' % savePath.split('/')[-1])
	plt.tight_layout()
	plt.xlabel('x-Coordinate (px)')
	plt.ylabel('y-Coordinate (px)')
	logging.info("Saving Image to file {}".format(output))
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


def prepareEvaluationNextStepCVCA(features):
	"""a helper function for evaluation.
	returns a pandas dataframe with CA and CV predictions for the next step, with the correct corresponding index"""

	xpredCV = []
	xpredCA = []
	ypredCV = []
	ypredCA = []
	
	for index, row in features.iterrows():
		xnextCV, ynextCV = predictionConstantVelNextStep(row.values)
		xnextCA, ynextCA = predictionConcentAccelNextStep(row.values)
		xpredCV.append(xnextCV)
		ypredCV.append(ynextCV)
		xpredCA.append(xnextCA)
		ypredCA.append(ynextCA)
	
	dataInp = {'CV_Prediction_X': xpredCV, 'CV_Prediction_Y': ypredCV,
			'CA_Prediction_X': xpredCA, 'CA_Prediction_Y': ypredCA}
	
	constantVelAndAccel = pd.DataFrame(data=dataInp, index=features.index)

	assert constantVelAndAccel['CV_Prediction_X'].head().iloc[0] == xpredCV[0]
	assert constantVelAndAccel['CV_Prediction_Y'].head().iloc[0] == ypredCV[0]
	assert constantVelAndAccel['CA_Prediction_X'].head().iloc[0] == xpredCA[0]
	assert constantVelAndAccel['CA_Prediction_Y'].head().iloc[0] == ypredCA[0]
	
	return constantVelAndAccel


def predictionConstantVelNextStep(array):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of x and y coordinate of a prediction made with the CV model"""
	
	assert len(array) >= 4  # assume featuresize >= 2
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	
	nextX = array[indexLastX] + v_x
	nextY = array[indexLastY] + v_y
	
	return nextX, nextY


def predictionConcentAccelNextStep(array):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of x and y coordinate of a prediction made with the CA model"""
	
	assert len(array) >= 6  # assume featuresize >= 3
	
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	a_x = v_x - (array[indexLastX - 1] - array[indexLastX - 2])
	a_y = v_y - (array[indexLastY - 1] - array[indexLastY - 2])
	
	t_delta = 1  # 1 time unit between observations
	nextX = array[indexLastX] + t_delta * v_x + 0.5 * t_delta ** 2 * a_x
	nextY = array[indexLastY] + v_y + a_y
	
	return nextX, nextY


def evaluateResultNextStep(X_test, y_test, totalPredictions):
	"""function to evaluate the result of this nets nextStep prediction.
	no return value, but writes a description of the error distribution and shows some plots for better visualisation"""
	
	xPredL = [p[0] for p in totalPredictions]
	yPredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'NNPredictionX': xPredL, 'NNPredictionY': yPredL}, index=y_test.index)
	
	caCvDf = prepareEvaluationNextStepCVCA(X_test)

	pandasLost = pd.concat([X_test, y_test, pandasLost, caCvDf], axis=1)
	
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
	
	
	# with open('pandasLost.pkl', 'wb') as f:
	#	pickle.dump(pandasLost, f)

	# logging.info(pandasLost['NNpixelErrorTotal'].head())
	
	relevantColumns = ['NNpixelErrorX', 'NNpixelErrorY', 'NNpixelErrorTotal',
							 'CVpixelErrorX', 'CVpixelErrorY', 'CVpixelErrorTotal',
							 'CApixelErrorX', 'CApixelErrorY', 'CApixelErrorTotal']
	
	reducedRelColumns = [relevantColumns[2], relevantColumns[5], relevantColumns[8]]
	
	logging.info("\n{}".format(pandasLost[reducedRelColumns].describe()))

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


# hist = pandasLost.hist(bins=10)
# plt.show()


def mirrorSingleFeature(points, midpoint, separator, direction=True):
	"""helper function for Data Augmentation instance - mirror input
	given an array of points and a line to mirror them on, returns an array of mirrored coordinates"""
	
	newpoints = points
	if not separator:
		if direction:
			relIndices = [i for i in range(int((len(points) - 2) / 2))]
			relIndices.append(len(points) - 2)
		else:
			relIndices = [i for i in range(int((len(points) - 2) / 2), len(points) - 2)]
			relIndices.append(len(points) - 1)
	else:
		if direction:
			relIndices = [i for i in range(int((len(points) - 2) / 2))]
			relIndices.append(len(points) - 2)
		else:
			relIndices = [i for i in range(int((len(points) - 2) / 2), len(points) - 2)]
			relIndices.append(len(points) - 2)
	
	# print(relIndices)
	for i in relIndices:
		if points[i] < midpoint:
			newpoints[i] = points[i] + 2 * abs(midpoint - points[i])
		else:
			newpoints[i] = points[i] - 2 * abs(midpoint - points[i])
	
	return newpoints


def augmentData(featuresTrain, labelsTrain, midpoint, separator, direction=True):
	"""applies the mirror input data augmentation to some data"""
	
	oldDf = pd.concat([featuresTrain, labelsTrain], axis=1, sort=False)
	newDf = oldDf.copy()
	newDf.index += oldDf.index.max()
	# newDf.apply(lambda x: pd.Series(mirrorSingleFeature(x, midpoint, separator, direction)), axis=1, result_type='broadcast')
	newDf.apply(lambda x: pd.Series(mirrorSingleFeature(x, midpoint, separator, direction)), axis=1,
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


def evaluateResultSeparator(X_test, y_test, totalPredictions):
	"""function to evaluate the result of this nets nextStep prediction.
	no return value, but writes a description of the error distribution and shows some plots for better visualisation"""
	
	xPredL = [p[0] for p in totalPredictions]
	timePredL = [p[1] for p in totalPredictions]
	pandasLost = pd.DataFrame(data={'NNPredictionPosBalken': xPredL, 'NNPredictionTime': timePredL}, index=y_test.index)
	
	caCvDf = prepareEvaluationNextStepCVCA(X_test)
	
	pandasLost = pd.concat([X_test, y_test, pandasLost, caCvDf], axis=1)
	
	pandasLost['NNpixelErrorPosBalken'] = pandasLost.apply(
		lambda row: (row['LabelPosBalken'] - row['NNPredictionPosBalken']), axis=1)
	pandasLost['NNerrorTime'] = pandasLost.apply(
		lambda row: (row['LabelTime'] - row['NNPredictionTime']), axis=1)
	
	relevantColumns = ['NNpixelErrorPosBalken', 'NNerrorTime']
	
	logging.info("\n{}".format(pandasLost[relevantColumns].describe()))

	# logging.info("number of predictions with error > 3: {}".format((pandasLost['NNpixelErrorTotal'] > 3).sum()))
	
	fig1, ax1 = plt.subplots()
	ax1.boxplot([pandasLost['NNpixelErrorPosBalken'], pandasLost['NNerrorTime']], showfliers=False)
	# plt.show()
	fig2, ax2 = plt.subplots()
	ax2.hist([pandasLost['NNpixelErrorPosBalken'], pandasLost['NNerrorTime']],
			 bins=40, label=['NNpixelErrorPosBalken', 'NNerrorTime'])
	# # plt.yscale('log')
	# ax.style.use('seaborn-muted')
	plt.legend(loc=1)
	plt.show()
