import tensorflow as tf
import numpy as np


def myCustomEstimator(features, labels, mode, params):
	"""Modell funktion für Custom Estimator (DNN Regression)"""

	# Input Layer
	top = tf.feature_column.input_layer(features, params["feature_columns"])

	activationFunc = params.get("activation", tf.nn.relu)
	# basierend auf hidden Units wird die Netztopologie aufgebaut
	arch = params.get("hidden_units", None)
	for units in arch:
		top = tf.layers.dense(inputs=top, units=units, activation=activationFunc)
		if "dropout" in params.keys() and params["dropout"] != 0:
			top = tf.layers.dropout(inputs=top, rate=params["dropout"], training=mode == tf.estimator.ModeKeys.TRAIN)

	# lineares output layer mit 2 Neuronen für die 2 Koordinaten
	output_layer = tf.layers.dense(inputs=top, units=2)
	# print(output_layer.shape)
	# Output layer umformen
	# predictions = tf.squeeze(output_layer, 2)

	if mode == tf.estimator.ModeKeys.PREDICT:
		# In `PREDICT` mode we only need to return predictions.
		return tf.estimator.EstimatorSpec(
			mode=mode, predictions={"predictions": output_layer})

	# Calculate loss using mean squared error
	
	scaleDim1 = params.get("scaleDim1")
	scaleDim2 = params.get("scaleDim2")
	weights_per_output = tf.constant([[scaleDim1, scaleDim2]])
	# print(weights_per_output.shape)

	average_loss = tf.losses.mean_squared_error(tf.cast(labels, tf.float32), output_layer, weights=weights_per_output)
	tf.summary.scalar("average_loss", average_loss)

	

	MSE = tf.metrics.mean_squared_error(tf.cast(labels, tf.float32), output_layer)
	tf.summary.scalar('error', MSE[1])
	
	l1Regularization = params.get("l1regularization", False) #Default: deactivate
	l2Regularization = params.get("l2regularization", False) #Default: deactivate
	
	regularization = l1Regularization or l2Regularization
	
	if regularization:
		if l1Regularization and l2Regularization:
			raise ValueError('L1 and L2 regularization are both activated')
		elif l1Regularization:
			regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
		elif l2Regularization:
			regularizer = tf.contrib.layers.l2_regularizer(scale=0.005, scope=None)
			
		trainVar = tf.trainable_variables()
		weights = [v for v in trainVar if "kernel" in v.name]
		
		assert len(weights) == (len(arch) + 1)
		
		regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
		regularized_loss = average_loss + regularization_penalty
	
	
	# Pre-made estimators use the total_loss instead of the average,
	# so report total_loss for compatibility.
	batch_size = tf.shape(labels)[0]
	total_loss = tf.to_float(batch_size) * average_loss

	if mode == tf.estimator.ModeKeys.TRAIN:
		decaying = params.get("decaying_learning_rate", False)
		if decaying:
			optimizer = params.get("optimizer", tf.train.AdamOptimizer)
			learningRate = tf.train.exponential_decay(
				learning_rate=params.get("learning_rate", None),
				global_step=tf.train.get_global_step(),
				decay_steps=params.get("decay_steps", 10000),
				decay_rate=0.96)
			optimizer = optimizer(learningRate)
		else:
			optimizer = params.get("optimizer", tf.train.AdamOptimizer)
			optimizer = optimizer(params.get("learning_rate", None))
		
		if not regularization:
			train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
		else:
			train_op = optimizer.minimize(loss=regularized_loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=total_loss, train_op=train_op)


	# In evaluation mode we will calculate evaluation metrics.
	assert mode == tf.estimator.ModeKeys.EVAL

	# Calculate root mean squared error
	rmse = tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float32), output_layer)

	# Add the rmse to the collection of evaluation metrics.
	eval_metrics = {"rmse": rmse, "average_loss": MSE}

	return tf.estimator.EstimatorSpec(
		mode=mode,
		# Report sum of error for compatibility with pre-made estimators
		loss=total_loss,
		eval_metric_ops=eval_metrics)