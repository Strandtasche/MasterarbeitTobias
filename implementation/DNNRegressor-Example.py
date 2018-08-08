#!/usr/bin/env python3

__author__ = "Tobias Hornberger"
__email__ = "saibot1207@googlemail.com"
__status__ = "Development"



# Importing necessary things
import tensorflow as tf
from tensorflow import estimator
import argparse
from starttf.utils.hyperparams import load_params
from tensorflow.python import debug as tf_debug

import pandas as pd

from MaUtil import *
import shutil
import numpy as np
import random
import sys

import os
import logging
from timeit import default_timer as timer

import loadDataExample as ld
import customEstimator as cE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('--training', help='if training of eval', action="store_true")
parser.add_argument('--plot', help='plotting with matplotlib', action="store_true")
parser.add_argument('--single', help='use all data in single set', action="store_true")
parser.add_argument('--fake', help="use real data?", action="store_true")
parser.add_argument('--plotNo', default=1, type=int, help="number of lines plotted")
parser.add_argument('--hyperparams', default="hyper_params.json", type=str, help="hyper parameter file to be used.")

# parser.add_argument('--save', help="store data", action="store_true")
parser.add_argument("--save", nargs='*', action="store", help="Store Data")
parser.add_argument("--load", nargs='*', action="store", help="load Data")
# parser.add_argument('--load', help="load stored data", action="store_true")

parser.add_argument('--dispWeights', help="display weights of neurons", action="store_true")
parser.add_argument('--separator', nargs='*', type=int, help='turn on prediction for separator')
parser.add_argument('--overrideModel', default=None, type=str, help="Model Path to override generated Path")
parser.add_argument('--overrideInput', default=None, type=str, help="input path to override input path from hyper_params")

parser.add_argument('--progressPlot', help="plot progress during training", action="store_true")
parser.add_argument('--debug', help="enable Debug mode", action="store_true")
parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
parser.add_argument('--lossAna', help="loss analysis", action="store_true")

parser.add_argument('--custom', help="use custom estimator", action="store_true")


def main(argv):
	args = parser.parse_args(argv[1:])

	TRAINING = args.training
	WITHPLOT = args.plot
	singleData = args.single
	FAKE = args.fake
	numberPrint = args.plotNo
	hyperParamFile = args.hyperparams
	saving = args.save
	loading = args.load
	overrideModelPath = args.overrideModel
	overrideInputPath = args.overrideInput
	usingCustomEstimator = args.custom

	displayWeights = args.dispWeights
	DEBUG = args.debug
	tensorboardDebugAddress = args.tensorboard_debug_address
	progressPlot = args.progressPlot

	maximumLossAnalysis = args.lossAna


	saveLoc = None
	if args.save is not None and args.load is not None:
		raise ValueError(
			"The --load and --save flags are mutually exclusive.")

	if args.save is not None and len(args.save) not in (0, 1):
		parser.error('Either give no values for save, or two, not {}.'.format(len(args.load)))
	elif args.save is not None:
		if len(args.save) == 0:
			# save to default location
			saveLoc = None
		elif len(args.save) == 1:
			# custom save location
			saveLoc = args.save[0]

	loadLoc = None
	if args.load is not None and len(args.load) not in (0, 1):
		parser.error('Either give no values for load, or one, not {}.'.format(len(args.load)))
		sys.exit(-1)
	elif args.load is not None:
		if len(args.load) == 0:
			# save to default location
			loadLoc = None
		elif len(args.load) == 1:
			# custom save location
			loadLoc = args.load[0]
			
	if args.separator is not None and FAKE:
		parser.error('No fake data for separator training (yet)')
			
	if args.separator is not None and len(args.separator) not in (0, 2):
		parser.error('Separator needs 2 Integers representing prediction Close off and separator position')
	elif args.separator is not None:
		separator = True
		if len(args.separator) == 0:
			separatorPosition = 1550
			predictionCutOff = 1300
		else:
			separatorPosition = args.separator[0]
			predictionCutOff = args.separator[1]
	else:
		separator = False
	

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
		if overrideInputPath is None:
			dataFolder = hyper_params.problem.data_path
		else:
			dataFolder = overrideInputPath
		testSize = hyper_params.data.testSize
		baseModelPath = hyper_params.problem.modelBasePath
		baseImagePath = hyper_params.problem.imagePath
	except AttributeError as err:
		logging.error("Error in Parameters. Maybe mistake in hyperparameter file?")
		logging.error("AttributeError: {0}".format(err))
		sys.exit(1)
	except Exception as e:
		logging.error("Some kind of error? not sure: {}".format(e))
		sys.exit(1)

	if loading is None:
		if not FAKE and not separator:
			# (X_train, y_train), (X_test, y_test) = ld.loadData(FEATURE_SIZE)
			(X_train, y_train), (X_test, y_test) = ld.loadRawMeasNextStep(dataFolder, FEATURE_SIZE, testSize)
		elif separator:
			(X_train, y_train), (X_test, y_test) = ld.loadRawMeasSeparation(dataFolder, FEATURE_SIZE, testSize, separatorPosition, predictionCutOff)
		else:
			(X_train, y_train), (X_test, y_test) = ld.loadFakeDataPandas(FEATURE_SIZE, FAKE_DATA_AMOUNT, testSize)
			
		#TODO: ziemlich hässlicher Hack - das könnte man noch schöner machen
		if singleData:
			X_train = pd.concat([X_train, X_test])
			X_test = X_train
			y_train = pd.concat([y_train, y_test])
			y_test = y_train

	# Network Design
	# --------------

	my_feature_columns = []
	columnNames = ld.genColumnNames(FEATURE_SIZE)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	if not overrideModelPath:
		MODEL_PATH = genModelPath(hyper_params, FAKE, usingCustomEstimator)
	else:
		MODEL_PATH = overrideModelPath

	logging.info('Saving to %s' % MODEL_PATH)

	if not usingCustomEstimator:
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
	else:
		test_config = estimator.RunConfig(save_checkpoints_steps=200,
		                                  save_checkpoints_secs=None)
		regressor = estimator.Estimator(
			model_fn=cE.myCustomEstimator,
			config=test_config,
			model_dir=MODEL_PATH,
			params={
				"feature_columns": my_feature_columns,
				"learning_rate": learningRate,
				"optimizer": tf.train.AdagradOptimizer,
				"hidden_units": hidden_layers,
				"dropout": dropout
			})

	if not os.path.exists(MODEL_PATH):
		os.makedirs(MODEL_PATH)
		logging.info("{} does not exist. Creating folder".format(MODEL_PATH))
	elif os.path.exists(MODEL_PATH) and not os.path.isdir(MODEL_PATH):
		logging.error("There is a file in the place where one would like to save their files..")
		sys.exit(1)

	if not os.path.exists(MODEL_PATH + '/' + os.path.basename(hyperParamFile)):
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(MODEL_PATH + hyperParamFile))
		# print("new hyperParam File written")
	else:
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(hyperParamFile)[:-5] + time_stamp + ".json")
		# print("added another version of hyper param file")

	if saving is not None:
		logging.info("storing data in data.h5")

		if saveLoc is None:
			saveLoc = MODEL_PATH + '/data.h5'

		with pd.HDFStore(saveLoc) as store:
			store['xtrain'] = X_train
			store['ytrain'] = y_train

			store['xtest'] = X_test
			store['ytest'] = y_test

	if loading is not None:
		try:
			if loadLoc is None:
				loadLoc = MODEL_PATH + '/data.h5'

			logging.info("loading data from store")

			with pd.HDFStore(loadLoc) as store:
				X_train = store['xtrain']
				y_train = store['ytrain']

				X_test = store['xtest']
				y_test = store['ytest']

		except Exception as e:
			logging.error("Error while loading from stored data: {}".format(e))
			sys.exit(1)

	#Plot progress Vars:
	if progressPlot:
		pos = [int(i * EPOCHS/10) for i in range(1, 10)]
		debugVisualizerIndex = random.randint(1, X_test.shape[0])
		featureVals = X_test.iloc[[debugVisualizerIndex]]
		labelVals = y_test.iloc[[debugVisualizerIndex]]
		predictions = []

	hooks = None


	if DEBUG and tensorboardDebugAddress:
		raise ValueError(
			"The --debug and --tensorboard_debug_address flags are mutually "
			"exclusive.")
	if DEBUG:
		hooks = [tf_debug.LocalCLIDebugHook()]


	# Start tensorboard with debugger port argument: "tensorboard --logdir=./debug2 --debugger_port 6007"
	elif tensorboardDebugAddress:
		hooks = [tf_debug.TensorBoardDebugHook(tensorboardDebugAddress)]
	# hooks = [debug_hook]

	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# Train it
	if TRAINING:
		logging.info('Train the DNN Regressor...\n')

		for epoch in range(EPOCHS):

			# Fit the DNNRegressor (This is where the magic happens!!!)
			# regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)
			regressor.train(input_fn=lambda: training_input_fn_Slices(X_train, y_train, BATCH_SIZE),
			                steps=STEPS_PER_EPOCH, hooks=hooks)

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

			if progressPlot and epoch in pos:
				debug_pred = regressor.predict(input_fn=lambda: eval_input_fn(featureVals, labels=None, batch_size=BATCH_SIZE))
				debug_predicted = [p['predictions'] for p in debug_pred]
				predictions.append(debug_predicted)

		if progressPlot:
			if FAKE:
				savePath = '/home/hornberger/testFake'
			else:
				savePath = '/home/hornberger/testReal'
			plotTrainDataPandas(featureVals, labelVals, predictions, savePath)

	# Now it's trained. We can try to predict some values.
	else:
		logging.info('No training today, just prediction')
		try:
			# Prediction
			eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(X_test, y_test, BATCH_SIZE))
			print('Error on whole Test set:\nMSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))
			averageLoss = eval_dict['average_loss']

		except ValueError as err:
			# probably failed to load model
			logging.error("{}".format(err))
			sys.exit(1)

		except Exception as e:
			logging.error("Unknown Error while trying to evaluate: {}".format(e))
			sys.exit(1)

		assert numberPrint < y_test.shape[0]

		sampleIndex = random.randint(0, y_test.shape[0] - numberPrint)

		x_pred2 = X_test.iloc[[sampleIndex + i for i in range(numberPrint)]]
		y_vals2 = y_test.iloc[[sampleIndex + i for i in range(numberPrint)]]

		print(x_pred2)
		print(y_vals2)
		
		startTime = timer()
		y_predGen = regressor.predict(input_fn=lambda: eval_input_fn(x_pred2, labels=None, batch_size=BATCH_SIZE))
		y_predicted = [p['predictions'] for p in y_predGen]
		endTime = timer()
		print("predicted: ")
		for i in y_predicted:
			print(i)
		print("time: {:.2f}s".format((endTime - startTime)))

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(x_pred2, y_vals2, 1))
		print('MSE (tensorflow): {0:f}'.format(eval_dict['average_loss']))

		if maximumLossAnalysis:
			totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(X_test, labels=None, batch_size=BATCH_SIZE))
			totalPredictions = [p['predictions'] for p in totalPredictGen]
			# for i in totalPredictions:
			# 	print(i)

			xPredL = [p[0] for p in totalPredictions]
			yPredL = [p[1] for p in totalPredictions]

			pandasLost = pd.DataFrame(data={'PredictionX': xPredL, 'PredictionY': yPredL}, index=y_test.index,
			                          columns=['PredictionX', 'PredictionY'])
			pandasLost = pd.concat([X_test, y_test, pandasLost], axis=1)
			pandasLost['MSE'] = pandasLost.apply(lambda row: ((row['LabelX'] - row['PredictionX'])**2 + (row['LabelY'] - row['PredictionY'])**2)/2, axis=1)

			maximumLossAnalysisCount = numberPrint #TODO: potenziell variable mit Arg? / separat?

			printDF = pandasLost.sort_values(by='MSE', ascending=False).head(maximumLossAnalysisCount)
			# print(printDF)
			plotDataPandas(maximumLossAnalysisCount, printDF[columnNames] ,printDF[['LabelX', 'LabelY']], printDF[['PredictionX', 'PredictionY']], baseImagePath,
			               baseImagePath + os.path.basename(MODEL_PATH) +'_'+ 'highestLoss' + '_' + time_stamp + '.png')

			# overAverageFeatures = []
			# overAverageLabels = []
			# overAverageLoss = []
			# overAveragePrediction = []
			# threshold = 10
			# for index, row in X_test.iterrows():
			# 	exampleFeatures = X_test.loc[[index]]
			# 	exampleLabel = y_test.loc[[index]]
			# 	# print(type(exampleFeatures))
			# 	# print(type(exampleLabel))
			# 	vali = regressor.evaluate(input_fn=lambda: eval_input_fn(exampleFeatures, exampleLabel, 1))
			# 	bad_pred = regressor.predict(input_fn=lambda: eval_input_fn(exampleFeatures, labels=None, batch_size=1))
			# 	bad_predicted = [p['predictions'] for p in bad_pred]
			# 	# print(vali)
			# 	if vali['average_loss'] > averageLoss:
			# 		overAverageFeatures.append(exampleFeatures)
			# 		overAverageLabels.append(exampleLabel)
			# 		overAverageLoss.append(vali['average_loss'])
			# 		overAveragePrediction.append(bad_predicted[0])
			# 		# print(exampleFeatures)
			# 		# print(exampleLabel)
			# 	if len(overAverageLabels) == threshold:
			# 		print("Length: {}".format(len(overAverageFeatures)))
			# 		threshold = threshold + 10
			#
			# tempDf1 = pd.concat(overAverageFeatures)
			# tempDf2 = pd.concat(overAverageLabels)
			#
			# finalDf = pd.concat([tempDf1, tempDf2], axis=1)
			# plotDataPandas(len(overAverageFeatures), tempDf1, tempDf2, overAveragePrediction, baseImagePath,
			#                baseImagePath + os.path.basename(MODEL_PATH) +'_'+ 'highestLoss' + '_' + time_stamp + '.png')
			#
			# print(finalDf)

		# displaying weights in Net - (a bit redundant after implementation of debugger
		if displayWeights:
			for variable in regressor.get_variable_names():
				print("name: \n{}\n\nvalue: \n{}".format(variable, regressor.get_variable_value(variable)))

		# # Final Plot
		if WITHPLOT:
			plotDataPandas(numberPrint, x_pred2, y_vals2, y_predicted, baseImagePath,
			               baseImagePath + os.path.basename(MODEL_PATH) + '_' + time_stamp + '.png')

	# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.basicConfig(level=logging.INFO)
	logging.info('Tensorflow %s' % tf.__version__)
	tf.app.run(main)
