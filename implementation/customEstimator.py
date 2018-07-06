import tensorflow as tf


def myCustomEstimator(features, labels, mode, params):
	"""Modell funktion für Custom Estimator (DNN Regression)"""

	# Input Layer
	top = tf.feature_column.input_layer(features, params["feature_columns"])

	# basierend auf hidden Units wird die Netztopologie aufgebaut
	for units in params.get("hidden_units", [20]):
		top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
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
			mode=mode, predictions={"prediction": output_layer})

	# Calculate loss using mean squared error
	# print(labels)
	# print(output_layer)

	average_loss = tf.losses.mean_squared_error(tf.cast(labels, tf.float32), output_layer)
	tf.summary.scalar("average_loss", average_loss)

	MSE = tf.metrics.mean_squared_error(tf.cast(labels, tf.float32), output_layer)
	tf.summary.scalar('error', MSE[1])

	# Pre-made estimators use the total_loss instead of the average,
	# so report total_loss for compatibility.
	batch_size = tf.shape(labels)[0]
	total_loss = tf.to_float(batch_size) * average_loss

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = params.get("optimizer", tf.train.AdamOptimizer)
		optimizer = optimizer(params.get("learning_rate", None))
		train_op = optimizer.minimize(
			loss=average_loss, global_step=tf.train.get_global_step())

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