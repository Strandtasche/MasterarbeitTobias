#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Tobias Hornberger"
__email__ = "saibot1207@googlemail.com"
__status__ = "Development"



# Importing necessary things
import tensorflow as tf
from tensorflow import estimator
import argparse
from starttf.utils.hyperparams import load_params
from tensorflow.python import debug as tf_debug

from MaUtil import *
import shutil
import numpy as np
import random
import sys

import os
import logging
import pickle
from timeit import default_timer as timer

import loadDataExample as ld
import customEstimator as cE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=5)

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('--training', help='if training of eval', action="store_true")
parser.add_argument('--plot', help='plotting with matplotlib', action="store_true")
parser.add_argument('--single', help='use all data in single set', action="store_true")
parser.add_argument('--fake', help="use real data?", action="store_true")
parser.add_argument('--plotNo', default=10, type=int, help="number of lines plotted")
parser.add_argument('--hyperparams', default="hyper_params.json", type=str, help="hyper parameter file to be used.")

parser.add_argument("--save", nargs='*', action="store", help="Store Data")
parser.add_argument("--load", nargs='*', action="store", help="load Data")

parser.add_argument('--dispWeights', help="display weights of neurons", action="store_true")
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

parser.add_argument('--separator', nargs='*', type=int, help='turn on prediction for separator')
parser.add_argument('--augment', help="activate dataAugmentation on training data", action="store_true")
parser.add_argument('--target', type=float, help="accuracy target.")


def main(argv):
	args = parser.parse_args(argv[1:])

	logging.info("Cmdline Input: {}".format(argv))

	TRAINING = args.training
	WITHPLOT = args.plot
	singleData = args.single
	FAKE = args.fake
	numberPrint = args.plotNo
	hyperParamFile = args.hyperparams
	saving = args.save
	loading = args.load
	augment = args.augment
	overrideModelPath = args.overrideModel
	overrideInputPath = args.overrideInput
	usingCustomEstimator = args.custom

	displayWeights = args.dispWeights
	DEBUG = args.debug
	tensorboardDebugAddress = args.tensorboard_debug_address
	progressPlot = args.progressPlot

	maximumLossAnalysis = args.lossAna
	cancelThreshold = args.target

	# MIDPOINT = 1123

	saveLoc = None
	if args.save is not None and args.load is not None:
		raise ValueError(
			"The --load and --save flags are mutually exclusive.")

	if args.save is not None and len(args.save) not in (0, 1):
		parser.error('Either give no values for save, or two, not {}.'.format(len(args.save)))
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
		parser.error('Separator needs 2 Integers representing prediction Close off and separator position: given {}'.format(len(args.separator)))
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

	if cancelThreshold is not None and not TRAINING:
		logging.warning("target parameter is not useful when not in training")


	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	try:
		hyper_params = load_params(hyperParamFile)

		STEPS_PER_EPOCH = hyper_params.train.steps_per_epoch
		EPOCHS = hyper_params.train.epochs
		BATCH_SIZE = hyper_params.train.batch_size

		FEATURE_SIZE = hyper_params.arch.feature_size
		ACTIVATION = hyper_params.arch.activation # "leaky_relu", "relu", "linear", TODO: "sigmoid", "tanh"
		dropout = hyper_params.arch.dropout_rate
		hidden_layers = hyper_params.arch.hidden_layers
		regularization = hyper_params.arch.regularization

		if regularization is None or regularization.lower() == "no":
			l1regularization = False
			l2regularization = False
		elif regularization.lower() == "l1":
			l1regularization = True
			l2regularization = False
		elif regularization.lower() == "l2":
			l1regularization = False
			l2regularization = True
		else:
			raise AttributeError('invalid string in hyper_params.arch.regularization')

		if FAKE:
			FAKE_DATA_AMOUNT = hyper_params.data.numberFakeLines
		if augment:
			MIDPOINT = hyper_params.data.augmentMidpoint
			MIRRORRANGE = hyper_params.data.augmentRange
		testSize = hyper_params.data.testSize
		limits = hyper_params.data.limits

		elementsDirection = hyper_params.data.direction
		if elementsDirection.lower() == "y":
			elementsDirectionBool = True
		elif elementsDirection.lower() == "x":
			elementsDirectionBool = False

		unitLocDirection = hyper_params.data.unitLoc
		unitTimeDirection = hyper_params.data.unitTime
		units = {'loc': unitLocDirection, 'time':unitTimeDirection}

		optimizer = hyper_params.train.optimizer # "Adam", "Adagrad"
		learningRate = hyper_params.train.learning_rate
		decaySteps = hyper_params.train.decay_steps

		if overrideInputPath is None:
			dataFolder = hyper_params.problem.data_path
		else:
			dataFolder = overrideInputPath

		baseModelPath = hyper_params.problem.modelBasePath
		baseImagePath = hyper_params.problem.imagePath
		if args.separator is None:
			if hyper_params.problem.separator == 1:
				separator = True
				separatorPosition = hyper_params.problem.separatorPosition
				predictionCutOff = hyper_params.problem.predictionCutOff
				thresholdPoint = hyper_params.problem.thresholdPoint
			else:
				separator = False

	except AttributeError as err:
		logging.error("Error in Parameters. Maybe mistake in hyperparameter file?")
		logging.error("AttributeError: {0}".format(err))
		sys.exit(1)
	except Exception as e:
		logging.error("Some kind of error? not sure: {}".format(e))
		sys.exit(1)



	if loading is None:
		if not FAKE and not separator:
			# (F_train, L_train), (F_test, L_test) = ld.loadData(FEATURE_SIZE)
			(F_train, L_train), (F_test, L_test), (labelMeans, labelStds) = ld.loadRawMeasNextStep(dataFolder, FEATURE_SIZE, testSize)
		elif separator:
			(F_train, L_train), (F_test, L_test), (labelMeans, labelStds) = ld.loadRawMeasSeparation(dataFolder, FEATURE_SIZE, testSize,
																			separatorPosition, predictionCutOff,
																			elementsDirectionBool)
		else:
			(F_train, L_train), (F_test, L_test) = ld.loadFakeDataPandas(FEATURE_SIZE, FAKE_DATA_AMOUNT, testSize)

		# TODO: ziemlich unschön - das könnte man noch besser machen
		if singleData:
			F_train = pd.concat([F_train, F_test])
			F_test = F_train
			L_train = pd.concat([L_train, L_test])
			L_test = L_train

		# ExTODO: find Augmentation MIDPOINT from data or as argument? - from Argument
		if augment:
			logging.info("applying augmentation to Training Set...")
			if separator:
				F_train, L_train = augmentData(F_train, L_train, MIDPOINT, MIRRORRANGE, separator, labelMeans, labelStds, direction=elementsDirectionBool)
			else:
				F_train, L_train = augmentData(F_train, L_train, MIDPOINT, MIRRORRANGE, separator, labelMeans, labelStds, direction=elementsDirectionBool)
			logging.info("done!")

	# Network Design
	# --------------

	my_feature_columns = []
	columnNames = ld.genColumnNames(FEATURE_SIZE)
	for key in columnNames:
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	if not overrideModelPath:
		MODEL_PATH = baseModelPath #genModelPath(hyper_params, FAKE, usingCustomEstimator, separator)
	else:
		MODEL_PATH = overrideModelPath

	logging.info("time: {}".format(time_stamp))
	logging.info('Saving to %s' % MODEL_PATH)

	if optimizer == 'Adagrad':
		opti = tf.train.AdagradOptimizer
	elif optimizer == 'Adam':
		opti = tf.train.AdamOptimizer
	# elif optimizer == 'GradientDescent':
	# 	opti = tf.train.GradientDescentOptimizer
	else:
		logging.error("No (or wrong) optimizer given in hyperparameter file")
		sys.exit(-1)

	if ACTIVATION == 'relu':
		acti = tf.nn.relu
	elif ACTIVATION == 'leaky_relu':
		acti = tf.nn.leaky_relu
	elif ACTIVATION == 'linear':
		acti = None
	else:
		logging.error("No (or wrong) activation function given in hyperparameter file")
		sys.exit(-1)

	if not os.path.exists(MODEL_PATH):
		os.makedirs(MODEL_PATH)
		logging.info("model folder {} does not exist. Creating folder".format(MODEL_PATH))
	elif os.path.exists(MODEL_PATH) and not os.path.isdir(MODEL_PATH):
		logging.error("There is a file in the place where one would like to save their files..")
		sys.exit(1)

	if not os.path.exists(baseImagePath):
		os.makedirs(baseImagePath)
		logging.info("image folder: {} does not exist. Creating folder".format(MODEL_PATH))

	if not os.path.exists(MODEL_PATH + '/' + os.path.basename(hyperParamFile)):
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(MODEL_PATH + hyperParamFile))
		# print("new hyperParam File written")
	else:
		shutil.copy2(hyperParamFile, MODEL_PATH + '/' + os.path.basename(hyperParamFile)[:-5] + time_stamp + ".json")
		# print("added another version of hyper param file")

	if saving is not None:
		logging.info("storing data in {}".format(saveLoc))

		if saveLoc is None:
			saveLoc = MODEL_PATH + '/data.h5'

		with pd.HDFStore(saveLoc) as store:
			store['xtrain'] = F_train
			store['ytrain'] = L_train

			store['xtest'] = F_test
			store['ytest'] = L_test

			store['labelMeans'] = labelMeans
			store['labelStds'] = labelStds

	if loading is not None:
		try:
			if loadLoc is None:
				loadLoc = MODEL_PATH + '/data.h5'

			logging.info("loading data from {}.".format(loadLoc))

			with pd.HDFStore(loadLoc) as store:
				F_train = store['xtrain']
				L_train = store['ytrain']

				F_test = store['xtest']
				L_test = store['ytest']

				labelMeans = store['labelMeans']
				labelStds = store['labelStds']

		except Exception as e:
			logging.error("Error while loading from stored data: {}".format(e))
			sys.exit(1)

	assert not F_train.index.duplicated().any()
	assert not L_train.index.duplicated().any()
	assert not F_test.index.duplicated().any()
	assert not L_test.index.duplicated().any()

	#Plot progress Vars:
	if progressPlot:
		pos = [int(i * EPOCHS/10) for i in range(1, 10)]
		debugVisualizerIndex = random.randint(1, F_test.shape[0])
		featureVals = F_test.iloc[[debugVisualizerIndex]]
		labelVals = L_test.iloc[[debugVisualizerIndex]]
		predictions = []

	if not usingCustomEstimator:
		# Validation and Test Configuration
		logging.info("using premade Estimator")
		test_config = estimator.RunConfig(save_checkpoints_steps=50000,
										  save_checkpoints_secs=None, save_summary_steps=100)

		regressor = estimator.DNNRegressor(feature_columns=my_feature_columns,
										   label_dimension=2,
										   hidden_units=hidden_layers,
										   model_dir=MODEL_PATH,
										   dropout=dropout,
										   activation_fn=acti,
										   config=test_config,
										   optimizer=opti(learning_rate=learningRate)
										   )
	else:
		logging.info("using custom estimator")
		test_config = estimator.RunConfig(save_checkpoints_steps=50000,
										  save_checkpoints_secs=None,
										  save_summary_steps=100)

		useRatioScaling = False # Todo: überlegen ob es hierfür noch eine sinnvolle verwendung gibt

		if separator and useRatioScaling:
			medianDim1 = L_train.iloc[:,0].median()
			medianDim2 = L_train.iloc[:,1].median()
			ratio = medianDim1 / medianDim2

			scaleDim1 = 1.0
			scaleDim2 = ratio
			logging.info("scaling loss between different dimensions. ScaleDim2-Ratio: {}".format(ratio))

		else:
			scaleDim1 = 1.0
			scaleDim2 = 1.0
		regressor = estimator.Estimator(
			model_fn=cE.myCustomEstimator,
			config=test_config,
			model_dir=MODEL_PATH,
			params={
				"feature_columns": my_feature_columns,
				"learning_rate": learningRate,
				"optimizer": opti,
				"hidden_units": hidden_layers,
				"dropout": dropout,
				"activation": acti,
				"decaying_learning_rate": True,
				"decay_steps": decaySteps,
				"l1regularization": l1regularization,
				"l2regularization": l2regularization,
				"scaleDim1": scaleDim1,
				"scaleDim2": scaleDim2,
				"regularizationStrength": 5e-08
			})

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

	logging.info("Train: ({}, {})".format(F_train.shape, L_train.shape))
	logging.info("Test: ({}, {})".format(F_test.shape, L_test.shape))
	logging.info("Means: \n{}".format(labelMeans))
	logging.info("Stds: \n{}".format(labelStds))


	# Train it
	if TRAINING:

		if not os.path.exists(MODEL_PATH + '/meanstd.pkl'):
			with open(MODEL_PATH + "/meanstd.pkl", 'wb') as f:
				pickle.dump([labelMeans, labelStds], f)
		else:
			with open(MODEL_PATH + "/meanstd.pkl", 'rb') as f:
				[labelMeansTemp, labelStdsTemp] = pickle.load(f)

				if not ((labelMeansTemp == labelMeans).all() and (labelStdsTemp == labelStds).all()): # does this work with float?
					logging.warning("CAREFUL: LabelMeans or LabelStds do not match existing values! Training with new values")

		logging.info('Train the DNN Regressor...\n')
		# test = tf.train.get_or_create_global_step()
		# logging.info("test: {}".format(test))

		epochInterm = []
		startTimeTraining = timer()

		for epoch in range(EPOCHS):

			# Fit the DNNRegressor (This is where the magic happens!!!)
			# regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)
			regressor.train(input_fn=lambda: training_input_fn_Slices(F_train, L_train, BATCH_SIZE),
			                steps=STEPS_PER_EPOCH, hooks=hooks)

			# Thats it -----------------------------
			# Start Tensorboard in Terminal:
			# 	tensorboard --logdir='./DNNRegressors/'
			# Now open Browser and visit localhost:6006\

			if epoch % 10 == 0:
				logging.info("Progress: epoch " + str(epoch))
				# logging.info("Progress: global step: {}".format(tf.train.get_global_step()))

				eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(F_test, L_test, BATCH_SIZE))
				logging.info("eval: " + str(eval_dict))

				avgLoss = eval_dict['average_loss']
				epochInterm.append(avgLoss)

				if cancelThreshold is not None:
					if avgLoss < cancelThreshold:
						logging.info("reached cancel Threshold. finishing training")
						break

			if progressPlot and epoch in pos:
				# TODO: adapt or remove because of standardize and normalize
				debug_pred = regressor.predict(input_fn=lambda: eval_input_fn(featureVals, labels=None, batch_size=BATCH_SIZE))
				debug_predicted = [p['predictions'] for p in debug_pred]
				predictions.append(debug_predicted)

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(F_test, L_test, BATCH_SIZE))

		logging.info("Training completed. final average loss: {}, best average loss during training: {}".format(
						eval_dict['average_loss'], min(epochInterm)))

		endTimeTraining = timer()
		timeTotal = endTimeTraining - startTimeTraining
		hours = timeTotal // 3600
		timeTotal %= 3600
		minutes = timeTotal // 60
		timeTotal %= 60
		logging.info("Total Training time: {}h {}min {}s".format(int(hours), int(minutes), int(timeTotal)))

		if progressPlot:
			if FAKE:
				savePath = '/home/hornberger/testFake'
			else:
				savePath = '/home/hornberger/testReal'
			plotTrainDataPandas(featureVals, labelVals, predictions, savePath, units)

	# Now it's trained. We can try to predict some values.
	else:
		logging.info('No training today, just prediction')

		if not os.path.exists(MODEL_PATH + '/meanstd.pkl'):
			logging.warning("Careful: No prior LabelMeans or LabelStds found!")
		else:
			with open(MODEL_PATH + "/meanstd.pkl", 'rb') as f:
				[labelMeansTemp, labelStdsTemp] = pickle.load(f)

				if not ((labelMeansTemp == labelMeans).all() and (labelStdsTemp == labelStds).all()): # does this work with float?
					logging.warning("evaluation on different dataset. replacing current labelMeans and labelStds")

					L_test = L_test * labelStds + labelMeans

					labelMeans = labelMeansTemp
					labelStds = labelStdsTemp

					logging.info("New labelMeans: \n{}".format(labelMeans))
					logging.info("New labelStds: \n{}".format(labelStds))

					L_test = (L_test - labelMeans) / labelStds



		try:
			# Prediction
			eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(F_test, L_test, BATCH_SIZE))
			logging.info('Error on whole Test set:\nMSE (tensorflow): {}'.format(eval_dict['average_loss']))
			averageLoss = eval_dict['average_loss']

		except ValueError as err:
			# probably failed to load model
			logging.error("{}".format(err))
			sys.exit(1)

		except Exception as e:
			logging.error("Unknown Error while trying to evaluate: {}".format(e))
			sys.exit(1)

		assert numberPrint < L_test.shape[0]

		sampleIndex = random.randint(0, L_test.shape[0] - numberPrint)

		# x_pred2 = F_test.iloc[[sampleIndex + i for i in range(numberPrint)]]
		# y_vals2 = L_test.iloc[[sampleIndex + i for i in range(numberPrint)]]

		x_pred2 = F_test.sample(n=numberPrint, random_state=sampleIndex)
		y_vals2 = L_test.sample(n=numberPrint, random_state=sampleIndex)
		y_vals2Denormalized = y_vals2.copy()
		for k in L_test.columns:
			y_vals2Denormalized[k] = y_vals2Denormalized[k] * labelStds[k] + labelMeans[k]

		print(x_pred2)
		print(y_vals2 * labelStds + labelMeans)

		startTime = timer()
		y_predGen = regressor.predict(input_fn=lambda: eval_input_fn(x_pred2, labels=None, batch_size=BATCH_SIZE))
		y_predicted = [p['predictions'] for p in y_predGen]
		endTime = timer()
		print("predicted: ")
		y_predictedCorr = [[x * b + c for x, b, c in zip(x, labelStds, labelMeans)] for x in y_predicted] # Look, ye mighty, and despair!
		for i in y_predictedCorr:
			print(i)
		print("time: {:.2f}s".format((endTime - startTime)))

		eval_dict = regressor.evaluate(input_fn=lambda: eval_input_fn(x_pred2, y_vals2, batch_size=BATCH_SIZE))
		print('MSE (tensorflow): {}'.format(eval_dict['average_loss']))

		if maximumLossAnalysis:
			if not separator:
				printDF = prepareMaximumLossAnalysisNextStep(F_test, L_test, numberPrint, regressor, BATCH_SIZE, labelMeans, labelStds)
				plotDataNextStepPandas(numberPrint, printDF[columnNames], printDF[['LabelX', 'LabelY']],
								   printDF[['PredictionX', 'PredictionY']], baseImagePath, limits, units,
								   os.path.basename(MODEL_PATH) + '_' + 'highestLoss' + '_' + time_stamp + '.png')
			else:
				printDF = prepareMaximumLossAnalysisSeparator(F_test, L_test, numberPrint, regressor, BATCH_SIZE, labelMeans, labelStds)
				# printDF['LabelPosBalken'] = printDF['LabelPosBalken'] * labelStds['LabelPosBalken'] + labelMeans['LabelPosBalken']
				plotDataSeparatorPandas(numberPrint, printDF[columnNames], printDF[['LabelPosBalken']],
										separatorPosition, printDF[['PredictionIntersect']], baseImagePath, limits, units, elementsDirectionBool,
										os.path.basename(MODEL_PATH) + '_' + 'highestLoss' + '_' + time_stamp + '.png')
			# print(printDF)

		# displaying weights in Net - (a bit redundant after implementation of debugger
		if displayWeights:
			for variable in regressor.get_variable_names():
				print("name: \n{}\nvalue: \n{}\n".format(variable, regressor.get_variable_value(variable)))

			weights = regressor.get_variable_value('dense/kernel')
			plt.imshow(weights, cmap='coolwarm')
			plt.show()

		# # Final Plot
		if WITHPLOT:
			L_trainDenormalized = L_train * labelStds + labelMeans
			L_testDenormalized = L_test * labelStds + labelMeans
			if not separator:
				plotDataNextStepPandas(numberPrint, x_pred2, y_vals2Denormalized, y_predictedCorr, baseImagePath, limits, units,
								   os.path.basename(MODEL_PATH) + '_' + time_stamp + '.png')

				totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(F_test, labels=None, batch_size=BATCH_SIZE))
				totalPredictions = [p['predictions'] for p in totalPredictGen]
				totalPredictionsCorr = [[x * b + c for x, b, c in zip(x, labelStds, labelMeans)] for x in totalPredictions] # Look, ye mighty, and despair!
				evaluateResultNextStep(F_test, L_testDenormalized, totalPredictionsCorr, units, baseImagePath)

			else:
				# y_vals2Denormalized = y_vals2['LabelPosBalken'] * labelStds['LabelPosBalken'] + labelMeans['LabelPosBalken']
				# y_predictedCorr = list(map(lambda x: [v * labelStds[k] + labelMeans[k] for k,v in enumerate(x)], y_predicted))

				plotDataSeparatorPandas(numberPrint, x_pred2, y_vals2Denormalized['LabelPosBalken'], separatorPosition,
										y_predictedCorr, baseImagePath, limits, units,  elementsDirectionBool,
										os.path.basename(MODEL_PATH) + '_' + time_stamp + '.png')
				totalPredictGen = regressor.predict(input_fn=lambda: eval_input_fn(F_test, labels=None, batch_size=BATCH_SIZE))
				totalPredictions = [p['predictions'] for p in totalPredictGen]
				totalPredictionsCorr = [[x * b + c for x, b, c in zip(x, labelStds, labelMeans)] for x in totalPredictions] # Look, ye mighty, and despair!

				filteredFeatures = filterDataForIntersection(F_train, thresholdPoint, elementsDirectionBool)
				medianAccel = getMedianAccel(filteredFeatures, separator, elementsDirectionBool)
				optimalAccel = getOptimalAccel(filteredFeatures, L_trainDenormalized.loc[filteredFeatures.index], separatorPosition, elementsDirectionBool)
				bias = getCVBias(filteredFeatures, L_trainDenormalized.loc[filteredFeatures.index], separatorPosition, elementsDirectionBool)

				configDict = {'medAc': medianAccel, 'optAc': optimalAccel, 'cvBias': bias}

				evaluateResultSeparator(F_test, L_testDenormalized, totalPredictionsCorr, separatorPosition, thresholdPoint,
										configDict, units, baseImagePath, elementsDirectionBool)


# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.basicConfig(level=logging.INFO)
	logging.info('Tensorflow %s' % tf.__version__)
	tf.app.run(main)
