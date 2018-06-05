#!/usr/bin/env python3

# Importing necessary things
import tensorflow as tf
from tensorflow import estimator as estimator
import argparse
from starttf.utils.hyperparams import load_params


import numpy as np
np.set_printoptions(precision=2)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time
import datetime

import logging

import loadDataExample as ld

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__)

TRAINING = False
WITHPLOT = False




(X_train, y_train), (X_test, y_test) = ld.loadData(5)

# X_train = np.random.random((15, 4))
# y_train = np.random.random((15,))
# X_test = X_train
# y_test = y_train


# Defining the Tensorflow input functions
# for training
def training_input_fn(batch_size=1):
	return tf.estimator.inputs.numpy_input_fn(x={ld.CSV_COLUMN_NAMES[k]: X_train[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))},
											  y=y_train.astype(np.float32),
											  batch_size=batch_size,
											  num_epochs=None,
											  shuffle=True)
# for test
def test_input_fn():
	return tf.estimator.inputs.numpy_input_fn(
		x={ld.CSV_COLUMN_NAMES[k]: X_test[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))},
		y=y_test.astype(np.float32),
		num_epochs=1,
		shuffle=False)


def training_input_fn_Slices(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	featureDict = {ld.CSV_COLUMN_NAMES[k]: features[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))}

	dataset = tf.data.Dataset.from_tensor_slices((featureDict, labels.astype(np.float32)))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.

	return dataset

def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	featureDict = {ld.CSV_COLUMN_NAMES[k]: features[:,k] for k in range(len(ld.CSV_COLUMN_NAMES))}
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

# Network Design
# --------------
#OLD: feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

my_feature_columns = []
for key in ld.CSV_COLUMN_NAMES:
	my_feature_columns.append(tf.feature_column.numeric_column(key=key))

hyper_params = load_params("hyper_params.json")
time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

STEPS_PER_EPOCH = hyper_params.train.steps_per_epoch
EPOCHS = hyper_params.train.epochs
BATCH_SIZE = hyper_params.train.batch_size

hidden_layers = [16, 16, 16, 16, 16]
dropout = hyper_params.arch.dropout_rate

MODEL_PATH='./DNNRegressors/'
for hl in hidden_layers:
	MODEL_PATH += '%s_' % hl
MODEL_PATH += 'D0%s' % (int(dropout*10))
logging.info('Saving to %s' % MODEL_PATH)

# Validation and Test Configuration
validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
test_config = estimator.RunConfig(save_checkpoints_steps=100,
								  save_checkpoints_secs=None)
# Building the Network
regressor = estimator.DNNRegressor(feature_columns=my_feature_columns,
								   label_dimension=1,
								   hidden_units=hidden_layers,
								   model_dir=MODEL_PATH,
								   dropout=dropout,
								   config=test_config)

# Train it
if TRAINING:
	logging.info('Train the DNN Regressor...\n')

	for epoch in range(EPOCHS+1):

		# Fit the DNNRegressor (This is where the magic happens!!!)
		#regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)
		regressor.train(input_fn=lambda :training_input_fn_Slices(X_train, y_train, BATCH_SIZE), steps=STEPS_PER_EPOCH)


		# Thats it -----------------------------
		# Start Tensorboard in Terminal:
		# 	tensorboard --logdir='./DNNRegressors/'
		# Now open Browser and visit localhost:6006\


		if epoch%10==0:
			print("we are making progress: " + str(epoch))


# Now it's trained. We can try to predict some values.
else:
# 	logging.info('No training today, just prediction')
# 	try:
		# Prediction
	eval_dict = regressor.evaluate(input_fn=lambda :eval_input_fn(X_test, y_test, 2))
	print("eval: " + str(eval_dict))
	# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	X_pred = np.linspace(0,1,11)
	y_pred = regressor.predict(input_fn=lambda :eval_input_fn(X_test, None, 1))
	print(y_pred)

	# Get trained values out of the Network
	for variable_name in regressor.get_variable_names():
		if str(variable_name).startswith('dnn/hiddenlayer') and \
				(str(variable_name).endswith('weights') or \
				 str(variable_name).endswith('biases')):
			print('\n%s:' % variable_name)
			weights = regressor.get_variable_value(variable_name)
			print(weights)
			print('size: %i' % weights.size)

		# # Final Plot
		# if WITHPLOT:
		# 	plt.plot(X, y, label='function to predict')
		# 	plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), \
		# 			 label='DNNRegressor prediction')
		# 	plt.legend(loc=2)
		# 	plt.ylim([0, 1])
		# 	plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
		# 	plt.tight_layout()
		# 	plt.savefig(MODEL_PATH + '.png', dpi=72)
		# 	plt.close()
	# except:
	# 	logging.error('Prediction failed! Maybe first train a model?')

