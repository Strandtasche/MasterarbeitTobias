#!/usr/bin/env python3

__author__ = "Tobias Hornberger"
__email__ = "saibot1207@googlemail.com"
__status__ = "Development"



# Importing necessary things
from tensorflow import estimator as estimator
import argparse
from starttf.utils.hyperparams import load_params
import pandas as pd

from MaUtil import *
import shutil
import numpy as np
import random

import os
import logging

import loadDataExample as ld

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()
parser.add_argument('--training', help='if training of eval', action="store_true")
parser.add_argument('--plot', help='plotting with matplotlib', action="store_true")
parser.add_argument('--fake', help="use real data?", action="store_true")
parser.add_argument('--plotNo', default=1, type=int, help="number of lines plotted")
parser.add_argument('--hyperparams', default="hyper_params.json", type=str, help="hyper parameter file to be used.")

parser.add_argument('--save', help="store data", action="store_true")
parser.add_argument('--load', help="load stored data", action="store_true")

parser.add_argument('--dispWeights', help="display weights of neurons", action="store_true")

parser.add_argument('--overwriteModel', default="", type=str, help="Model Path to overwrite generated Path")

parser.add_argument('--debug', help="enable Debug mode", action="store_true")


def main(argv):

	args = parser.parse_args(argv[1:])

	TRAINING = args.training
	WITHPLOT = args.plot
	FAKE = args.fake
	numberPrint = args.plotNo
	hyperParamFile = args.hyperparams
	saving = args.save
	loading = args.load
	overWriteModelPath = args.overwriteModel

	displayWeights = args.dispWeights

	DEBUG = args.debug

	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

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

	if not loading:
		if not FAKE:
			# (X_train, y_train), (X_test, y_test) = ld.loadData(FEATURE_SIZE)
			(X_train, y_train), (X_test, y_test) = ld.loadRawMeas(dataFolder, FEATURE_SIZE, testSize)
		else:
			(X_train, y_train), (X_test, y_test) = ld.loadFakeDataPandas(FEATURE_SIZE, FAKE_DATA_AMOUNT, testSize)

	# Network Design
	# --------------
	# OLD: feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

	my_feature_columns = []
	columnNames = ld.genColumnNames(FEATURE_SIZE)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	if len(overWriteModelPath) == 0:
		if not FAKE:
			MODEL_PATH = baseModelPath
		else:
			MODEL_PATH = baseModelPath[0:-1] + "FakeData/"

		for hl in hidden_layers:
			MODEL_PATH += '%s_' % hl
		MODEL_PATH += 'D0%s_' % (int(dropout * 100))
		MODEL_PATH += 'FS%s_' % (FEATURE_SIZE)
		MODEL_PATH += 'LR%s_' % (str(learningRate % 1)[2:])

		if FAKE:
			MODEL_PATH += 'FD%s' % (FAKE_DATA_AMOUNT)
		else:
			MODEL_PATH += 'DS_%s' % (dataFolder.replace("/", "_"))
	else:
		MODEL_PATH = overWriteModelPath

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

	# if (loading or saving) and FAKE:
	# 	logging.error("no pickling of fake data. Sorry")
	# 	exit()

	if not os.path.exists(MODEL_PATH):
		os.makedirs(MODEL_PATH)
		logging.info("{} does not exist. Creating folder".format(MODEL_PATH))
	elif os.path.exists(MODEL_PATH) and not os.path.isdir(MODEL_PATH):
		logging.warning("There is a file in the place where one would like to save their files..")

	if not os.path.exists(MODEL_PATH + '/' + os.path.basename(hyperParamFile)):
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(MODEL_PATH + hyperParamFile))
		# print("new hyperParam File written")
	else:
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(hyperParamFile)[:-5] + time_stamp + ".json")
		# print("added another version of hyper param file")


	if saving: #and not FAKE:
		# logging.info("pickling Data to Model Path")

		# X_train.to_pickle(MODEL_PATH + '/X_train.pkl')
		# y_train.to_pickle(MODEL_PATH + '/y_train.pkl')
		#
		# X_test.to_pickle(MODEL_PATH + '/X_test.pkl')
		# y_test.to_pickle(MODEL_PATH + '/y_test.pkl')

		logging.info("storing data in data.h5")

		with pd.HDFStore(MODEL_PATH + '/data.h5') as store:
			store['xtrain'] = X_train
			store['ytrain'] = y_train

			store['xtest'] = X_test
			store['ytest'] = y_test

	if loading: # and not FAKE:
		try:
			# logging.info("loading pickled data")
			# X_train = pd.read_pickle(MODEL_PATH + '/X_train.pkl')
			# y_train = pd.read_pickle(MODEL_PATH + '/y_train.pkl')
			#
			# X_test = pd.read_pickle(MODEL_PATH + '/X_test.pkl')
			# y_test = pd.read_pickle(MODEL_PATH + '/y_test.pkl')

			logging.info("loading data from store")

			with pd.HDFStore(MODEL_PATH + '/data.h5') as store:
				X_train = store['xtrain']
				y_train = store['ytrain']

				X_test = store['xtest']
				y_test = store['ytest']

		except:
			# logging.error("Error while loading from pickled data")
			logging.error("Error while loading from stored data")

	###DEBUG Vars:
	if DEBUG:
		pos = [int(i * EPOCHS/10)  for i in range(1, 10)]
		debugVisualizerIndex = random.randint(1, X_test.shape[0])
		featureVals = X_test.iloc[[debugVisualizerIndex]]
		labelVals = y_test.iloc[[debugVisualizerIndex]]
		predictions = []

	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

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

			if DEBUG and epoch in pos:
				debug_pred = regressor.predict(input_fn=lambda: eval_input_fn(featureVals, labels=None, batch_size=BATCH_SIZE))
				debug_predicted = [p['predictions'] for p in debug_pred]
				predictions.append((debug_predicted))


		if DEBUG:
			if FAKE:
				savePath = '/home/hornberger/testFake'
			else:
				savePath = '/home/hornberger/testReal'
			plotTrainDataPandas(featureVals, labelVals, predictions, savePath)

	# Now it's trained. We can try to predict some values.
	else:
		# 	logging.info('No training today, just prediction')
		# 	try:
		# Prediction
		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test, y_test, BATCH_SIZE))
		print('Error on whole Test set:\nMSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		assert numberPrint < y_test.shape[0]

		sampleIndex = random.randint(0, y_test.shape[0] - numberPrint)

		# if FAKE: #X_test is a dict of ndarrays
		#
		# 	x_pred2 = {}
		# 	for i in columnNames:
		# 		x_pred2[i] = [X_test[i].item(sampleIndex + k) for k in range(numberPrint)]
		#
		# 	y_vals2 = np.array([y_test[sampleIndex + k] for k in range(numberPrint)])
		#
		# else: #X_test is a pandas dataframe
		x_pred2 = X_test.iloc[[sampleIndex + i for i in range(numberPrint)]]
		y_vals2 = y_test.iloc[[sampleIndex + i for i in range(numberPrint)]]

		print(x_pred2)
		print(y_vals2)

		y_pred = regressor.predict(input_fn=lambda: eval_input_fn(x_pred2, labels=None, batch_size=BATCH_SIZE))
		y_predicted = [p['predictions'] for p in y_pred]
		print("predicted: ")
		# print(y_predicted)
		for i in y_predicted:
			print(i)

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(x_pred2, y_vals2, 1))
		print('MSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		if displayWeights:
			for variable in regressor.get_variable_names():
				print("name: \n{}\n value: \n{}".format(variable, regressor.get_variable_value(variable)))

		# # Final Plot
		if WITHPLOT:
			# if type(x_pred2) == dict:
			# 	plotDataNumpy(numberPrint, x_pred2, y_vals2, y_predicted, MODEL_PATH)
			# else:
			plotDataPandas(numberPrint, x_pred2, y_vals2, y_predicted, MODEL_PATH)

	# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.basicConfig(level=logging.INFO)
	logging.info('Tensorflow %s' % tf.__version__)
	tf.app.run(main)
