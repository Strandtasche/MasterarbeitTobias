#
# Based on script from https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
#
# Modified
#
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
	return np.cos(1.5 * np.pi * X)

np.random.seed(3)

n_samples = 45
degrees = [1, 4, 17]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.2
titles = ['Underfitting', 'passendes Modell', 'Overfitting']

plt.figure() #figsize=(14, 5))
for i in range(len(degrees)):
	fig, ax = plt.subplots() # 1, len(degrees), i + 1)
	plt.setp(ax, xticks=(), yticks=())

	polynomial_features = PolynomialFeatures(degree=degrees[i],
											 include_bias=False)
	linear_regression = LinearRegression()
	pipeline = Pipeline([("polynomial_features", polynomial_features),
						 ("linear_regression", linear_regression)])
	pipeline.fit(X[:, np.newaxis], y)

	# Evaluate the models using crossvalidation
	scores = cross_val_score(pipeline, X[:, np.newaxis], y,
							 scoring="neg_mean_squared_error", cv=10)

	X_test = np.linspace(0, 1, 100)
	plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
	plt.plot(X_test, true_fun(X_test), label="True function")
	plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
	plt.xlabel("Time")
	plt.ylabel("Value")
	plt.xlim((0, 1))
	plt.ylim((-2, 2))
	plt.legend(loc=1)
	plt.show()
	fig.savefig('/home/hornberger/Pictures/plot' + titles[i] + '.png', dpi=900)
# plt.title(titles[i])

plt.show()