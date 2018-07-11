import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time
import logging
from adjustText import adjust_text

def genModelPath(hyperparams, fake, usingCustomestimator):
	"""returns path to location of model based on parameters"""
	if not fake:
		MODEL_PATH = hyperparams.problem.modelBasePath
	else:
		MODEL_PATH = hyperparams.problem.modelBasePath[0:-1] + "FakeData/"

	if usingCustomestimator:
		MODEL_PATH += 'CustomEstimator_'
	else:
		MODEL_PATH += 'PremadeEstimator_'

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
		plt.plot(y_vals2[k][0], y_vals2[k][1], 'go', label='target')
		plt.plot(y_predicted[k][0], y_predicted[k][1], 'bo', label='prediction')
	plt.plot()
	#plt.legend(loc=9)

	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	plt.savefig(MODEL_PATH + '_' + time_stamp + '.png', dpi=300)
	# plt.show()
	plt.close()


def plotDataPandas(numberPrint, x_pred2, y_vals2, y_predicted, savePath, name=None):
	"""plot a certain number of examples (from pandas dataframe)
	with features, labels and prection and save to .png file"""
	MODEL_PATH = savePath
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	if name is None:
		output = MODEL_PATH + 'Pic_' + time_stamp + '.png'
	else:
		output = name

	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].head(numberPrint).values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].head(numberPrint).values
	x_t = y_vals2[['LabelX']].head(numberPrint).values
	y_t = y_vals2[['LabelY']].head(numberPrint).values

	# print(x)
	# print(y)

	assert len(x) == len(y)

	plt.plot(x, y, 'ro',label='function to predict')
	plt.plot(x_t, y_t, 'go', label='target')
	if isinstance(y_predicted, list):
		for k in range(numberPrint):
			plt.plot(y_predicted[k][0], y_predicted[k][1], 'bo', label='prediction')
	else:
		for k in range(numberPrint):
			plt.plot(y_predicted['PredictionX'], y_predicted['PredictionY'], 'bo', label='prediction')
	plt.plot()

	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	logging.info("Saving Image to file {}".format(output))
	plt.savefig(output, dpi=300)
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
	plt.plot(x_t, y_t, 'go', label='target')
	i = 0
	total = len(y_predicted)
	textArray = []
	for elem in y_predicted:
		plt.plot(elem[0][0], elem[0][1], 'bo', label='prediction')
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
