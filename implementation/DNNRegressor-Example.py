#!/usr/bin/env python3

# Importing necessary things
import tensorflow as tf
from tensorflow import estimator as estimator
import loadDataExample as ld

import numpy as np
np.set_printoptions(precision=2)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import logging

import loadDataExample as ld

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__)

TRAINING = True
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





# Network Design
# --------------
#OLD: feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

my_feature_columns = []
for key in ld.CSV_COLUMN_NAMES:
	my_feature_columns.append(tf.feature_column.numeric_column(key=key))

STEPS_PER_EPOCH = 100
EPOCHS = 200
BATCH_SIZE = 100

hidden_layers = [16, 16, 16, 16, 16]
dropout = 0.0

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
		regressor.train(input_fn=training_input_fn(batch_size=BATCH_SIZE), steps=STEPS_PER_EPOCH)


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
	eval_dict = regressor.evaluate(input_fn=test_input_fn())
	print("eval: " + str(eval_dict))
	# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	X_pred = np.linspace(0,1,11)
	y_pred = regressor.predict(input_fn=test_input_fn())
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

