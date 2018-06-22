#!/usr/bin/env python3

# Importing necessary things
import tensorflow as tf
from tensorflow import estimator as estimator
import argparse
from starttf.utils.hyperparams import load_params

from MaUtil import *

import numpy as np


import os
import logging

import loadDataExample as ld

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()
parser.add_argument('--training', default=False, type=bool, help='if training of eval')
parser.add_argument('--plot', default=False, type=bool, help='plotting with matplotlib')
parser.add_argument('--fake', default=False, type=bool, help="use real data?")
parser.add_argument('--plotNo', default=1, type=int, help="number of lines plotted")
parser.add_argument('--hyperparams', default="hyper_params.json", type=str, help="hyper parameter file to be used.")


def main(argv):
	args = parser.parse_args(argv[1:])

	TRAINING = args.training
	WITHPLOT = args.plot
	FAKE = args.fake
	numberPrint = args.plotNo
	hyperParamFile = args.hyperparams

	try:
		hyper_params = load_params(hyperParamFile)
		STEPS_PER_EPOCH = hyper_params.train.steps_per_epoch
		EPOCHS = hyper_params.train.epochs
		BATCH_SIZE = hyper_params.train.batch_size
		FEATURE_SIZE = hyper_params.arch.feature_size
		FAKE_DATA_AMOUNT = hyper_params.data.numberFakeLines
		hidden_layers = hyper_params.arch.hidden_layers
		dropout = hyper_params.arch.dropout_rate
		learningRate = hyper_params.train.learning_rate
		dataFolder = hyper_params.problem.data_path
		testSize = hyper_params.data.testSize
		baseModelPath = hyper_params.problem.modelBasePath
	except AttributeError as err:
		logging.error("Error in Parameters. Maybe mistake in hyperparameter file?")
		logging.error("AttributeError: {0}".format(err))
		exit()
	except:
		logging.error("Some kind of error? not sure")
		exit()


	if not FAKE:
		# (X_train, y_train), (X_test, y_test) = ld.loadData(FEATURE_SIZE)
		(X_train, y_train), (X_test, y_test) = ld.loadRawMeas(dataFolder, FEATURE_SIZE, testSize)

	else:
		(X_train, y_train), (X_test, y_test) = ld.loadFakeData(FEATURE_SIZE, FAKE_DATA_AMOUNT, testSize)

	# Network Design
	# --------------
	# OLD: feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

	my_feature_columns = []
	columnNames = ld.genColumnNames(FEATURE_SIZE)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))


	if not FAKE:
		MODEL_PATH = baseModelPath
	else:
		MODEL_PATH = baseModelPath[0:-1] + "FakeData/"

	for hl in hidden_layers:
		MODEL_PATH += '%s_' % hl
	MODEL_PATH += 'D0%s_' % (int(dropout * 100))
	MODEL_PATH += 'FS%s_' % (FEATURE_SIZE)
	MODEL_PATH += 'LR%s_' % (int(learningRate * 100))

	if FAKE:
		MODEL_PATH += 'FD%s' % (FAKE_DATA_AMOUNT)
	else:
		MODEL_PATH += 'DS_%s' % (dataFolder.replace("/", "_"))

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
	                                   optimizer=tf.train.AdagradOptimizer(learning_rate=learningRate),
	                                   config=test_config)

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

				# eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_train, y_train, BATCH_SIZE))
				# print("eval: " + str(eval_dict))

	# Now it's trained. We can try to predict some values.
	else:
		# 	logging.info('No training today, just prediction')
		# 	try:
		# Prediction
		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test, y_test, BATCH_SIZE))
		print('Error on whole Test set:\nMSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		assert numberPrint < y_test.shape[0]

		if FAKE: #X_test is a dict of ndarrays

			x_pred2 = {}
			for i in columnNames:
				x_pred2[i] = [X_test[i].item(k) for k in range(numberPrint)]

			y_vals2 = np.array([y_test[i] for i in range(numberPrint)])

		else: #X_test is a pandas dataframe
			x_pred2 = X_test.head(numberPrint)
			y_vals2 = y_test.head(numberPrint)

		print(x_pred2)
		print(y_vals2)

		y_pred = regressor.predict(input_fn=lambda: eval_input_fn(x_pred2, labels=None, batch_size=1))
		y_predicted = [p['predictions'] for p in y_pred]
		print("predicted: ")
		# print(y_predicted)
		for i in y_predicted:
			print(i)

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(x_pred2, y_vals2, 1))
		print('MSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		# # Final Plot
		if WITHPLOT:
			if type(x_pred2) == dict:
				plotDataNumpy(numberPrint, x_pred2, y_vals2, y_predicted, MODEL_PATH)
			else:
				plotDataPandas(numberPrint, x_pred2, y_vals2, y_predicted, MODEL_PATH)
	# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.basicConfig(level=logging.INFO)
	logging.info('Tensorflow %s' % tf.__version__)
	tf.app.run(main)
