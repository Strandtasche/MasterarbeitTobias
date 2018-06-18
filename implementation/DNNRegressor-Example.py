#!/usr/bin/env python3

# Importing necessary things
import tensorflow as tf
from tensorflow import estimator as estimator
import argparse
from starttf.utils.hyperparams import load_params
import itertools

import numpy as np

np.set_printoptions(precision=2)

from sklearn import metrics
from sklearn import preprocessing


import matplotlib.pyplot as plt
import time
import os
import datetime
import shutil
import logging

import loadDataExample as ld

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--training', default=False, type=bool, help='if training of eval')
parser.add_argument('--plot', default=False, type=bool, help='plotting with matplotlib')
parser.add_argument('--fake', default=False, type=bool, help="use real data?")


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


def main(argv):
	args = parser.parse_args(argv[1:])

	TRAINING = args.training
	WITHPLOT = args.plot
	FAKE = args.fake

	try:
		hyper_params = load_params("hyper_params.json")
		STEPS_PER_EPOCH = hyper_params.train.steps_per_epoch
		EPOCHS = hyper_params.train.epochs
		BATCH_SIZE = hyper_params.train.batch_size
		FEATURE_SIZE = hyper_params.arch.feature_size
		FAKE_DATA_AMOUNT = hyper_params.data.numberFakeLines
		hidden_layers = hyper_params.arch.hidden_layers
		dropout = hyper_params.arch.dropout_rate
		learningRate = hyper_params.train.learning_rate
	except AttributeError as err:
		logging.error("Error in Parameters. Maybe mistake in hyperparameter file?")
		logging.error("AttributeError: {0}".format(err))
		exit()
	except:
		logging.error("Some kind of error? not sure")
		exit()

	if not FAKE:
		(X_train, y_train), (X_test, y_test) = ld.loadData(FEATURE_SIZE, FAKE_DATA_AMOUNT)
	else:
		(X_train, y_train), (X_test, y_test) = ld.loadFakeData(FEATURE_SIZE)

	# Network Design
	# --------------
	# OLD: feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

	my_feature_columns = []
	columnNames = ld.genColumnNames(FEATURE_SIZE)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	if not FAKE:
		MODEL_PATH = './DNNRegressors/'
	else:
		MODEL_PATH = './DNNRegressorsFAKE2/'

	for hl in hidden_layers:
		MODEL_PATH += '%s_' % hl
	MODEL_PATH += 'D0%s_' % (int(dropout * 10))
	MODEL_PATH += 'FS%s_' % (FEATURE_SIZE)

	if FAKE:
		MODEL_PATH += 'FD%s' % (FAKE_DATA_AMOUNT)

	logging.info('Saving to %s' % MODEL_PATH)

	# Validation and Test Configuration
	test_config = estimator.RunConfig(save_checkpoints_steps=200,
	                                  save_checkpoints_secs=None)
	# Building the Network
	regressor = estimator.DNNRegressor(feature_columns=my_feature_columns,
	                                   label_dimension=2,
	                                   hidden_units=hidden_layers,
	                                   model_dir=MODEL_PATH,
	                                   dropout=dropout,
	                                   optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate,
	                                                                               l1_regularization_strength=0.001),
	                                   config=test_config)

	try:
		shutil.copy2("./hyper_params.json", MODEL_PATH + '/hyper_params_' + time_stamp + ".json")
	except:
		print("params not saved in folder.")

	# Train it
	if TRAINING:
		logging.info('Train the DNN Regressor...\n')

		for epoch in range(EPOCHS):

			# Fit the DNNRegressor (This is where the magic happens!!!)
			# regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)
			regressor.train(input_fn=lambda: training_input_fn_Slices(X_train, y_train, BATCH_SIZE),
			                steps=STEPS_PER_EPOCH)

			# Thats it -----------------------------
			# Start Tensorboard in Terminal:
			# 	tensorboard --logdir='./DNNRegressors/'
			# Now open Browser and visit localhost:6006\

			if epoch % 10 == 0:
				print("we are making progress: " + str(epoch))

				eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test, y_test, BATCH_SIZE))
				print("eval: " + str(eval_dict))

	# Now it's trained. We can try to predict some values.
	else:
		# 	logging.info('No training today, just prediction')
		# 	try:
		# Prediction
		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test, y_test, BATCH_SIZE))
		print('MSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))
		# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

		# x_pred = {}
		# vals = [100 + 1*i for i in range(10)]
		# iter = 0
		# for i in columnNames:
		# 	x_pred[i] = [vals[iter]]
		# 	iter = iter + 1

		print(y_test.shape)

		x_pred2 = {}
		for i in columnNames:
			x_pred2[i] = [X_test[i].item(0)]

		y_vals2 = np.array([y_test[0]])

		print(x_pred2)
		print(y_vals2[0])

		y_pred = regressor.predict(input_fn=lambda: eval_input_fn(x_pred2, labels=None, batch_size=1))
		# predictions = list(p["predictions"] for p in itertools.islice(y_pred, 6))
		y_predicted = [p['predictions'] for p in y_pred][0]
		# y_predicted = y_predicted.reshape(np.array(y_test).shape)
		print("predicted: ")
		print(y_predicted)

		score_sklearn = metrics.mean_squared_error([y_predicted], y_vals2)
		print('MSE (sklearn): {0:f}'.format(score_sklearn))

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(x_pred2, y_vals2, 1))
		print('MSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		# # Final Plot
		if WITHPLOT:
			x = []
			y = []
			for i in x_pred2.keys():
				if i[0] == 'X':
					x.append(x_pred2[i][0])
				else:
					y.append(x_pred2[i][0])

			assert len(x) == len(y)



			plt.plot(x, y, 'ro',label='function to predict')
			plt.plot(y_test[0][0], y_test[0][1], 'go', label='target')
			# print(pred)
			plt.plot(y_predicted[0], y_predicted[1], 'bo', label='prediction')
			plt.plot()
			# plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), label='DNNRegressor prediction')
			plt.legend(loc=9)
			# plt.ylim([0, 1])

			plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
			plt.tight_layout()
			plt.savefig(MODEL_PATH + '_' + time_stamp +  '.png', dpi=72)
			plt.close()
	# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.basicConfig(level=logging.INFO)
	logging.info('Tensorflow %s' % tf.__version__)
	tf.app.run(main)
