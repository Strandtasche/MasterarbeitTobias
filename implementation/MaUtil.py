import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time


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
	# plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), label='DNNRegressor prediction')
	#plt.legend(loc=9)
	# plt.ylim([0, 1])

	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	plt.savefig(MODEL_PATH + '_' + time_stamp + '.png', dpi=72)
	plt.show()
	# plt.close()



def plotDataPandas(numberPrint, x_pred2, y_vals2, y_predicted, savePath):

	MODEL_PATH = savePath
	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	x = x_pred2[[i for i in x_pred2.columns if i[0] == 'X']].head(numberPrint).values
	y = x_pred2[[i for i in x_pred2.columns if i[0] == 'Y']].head(numberPrint).values
	x_t = y_vals2[['LabelX']].head(numberPrint).values
	y_t = y_vals2[['LabelY']].head(numberPrint).values

	# print(x)
	# print(y)

	assert len(x) == len(y)

	plt.plot(x, y, 'ro',label='function to predict')
	plt.plot(x_t, y_t, 'go', label='target')
	for k in range(numberPrint):
		plt.plot(y_predicted[k][0], y_predicted[k][1], 'bo', label='prediction')
	plt.plot()

	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
	plt.tight_layout()
	plt.savefig(MODEL_PATH + '_' + time_stamp + '.png', dpi=72)
	plt.show()
	# plt.close()




