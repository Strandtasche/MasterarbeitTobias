#Testfile
import logging

import numpy as np
import pandas as pd


def prepareEvaluationNextStepCVCA(features):
	"""a helper function for evaluation.
	returns a pandas dataframe with CA and CV predictions for the next step, with the correct corresponding index"""

	xpredCV = []
	xpredCA = []
	ypredCV = []
	ypredCA = []
	
	for index, row in features.iterrows():
		xnextCV, ynextCV = predictionConstantVelNextStep(row.values)
		xnextCA, ynextCA = predictionConstantAccelNextStep(row.values)
		xpredCV.append(xnextCV)
		ypredCV.append(ynextCV)
		xpredCA.append(xnextCA)
		ypredCA.append(ynextCA)
	
	dataInp = {'CV_Prediction_X': xpredCV, 'CV_Prediction_Y': ypredCV,
			'CA_Prediction_X': xpredCA, 'CA_Prediction_Y': ypredCA}
	
	constantVelAndAccel = pd.DataFrame(data=dataInp, index=features.index)

	assert constantVelAndAccel['CV_Prediction_X'].head().iloc[0] == xpredCV[0]
	assert constantVelAndAccel['CV_Prediction_Y'].head().iloc[0] == ypredCV[0]
	assert constantVelAndAccel['CA_Prediction_X'].head().iloc[0] == xpredCA[0]
	assert constantVelAndAccel['CA_Prediction_Y'].head().iloc[0] == ypredCA[0]
	
	return constantVelAndAccel


def prepareEvaluationSeparatorCVCA(features, medianAccel, separatorPosition, direction):
	"""a helper function for evaluation.
	returns a pandas dataframe with CA and CV predictions for separation, with the correct corresponding index"""
	
	locPredCV = []
	locPredCA = []
	locPredAA = []
	timePredCV = []
	timePredCA = []
	timePredAA = []
	
	for index, row in features.iterrows():
		locSepCV, timeSepCV = predictionConstantVelSeparator(row.values, separatorPosition, direction)
		locSepCA, timeSepCA = predictionConstantAccelSeparator(row.values, separatorPosition, direction)
		locSepAA, timeSepAA = predictionAverageAccelSeparator(row.values, medianAccel, separatorPosition, direction)
		locPredCV.append(locSepCV)
		timePredCV.append(timeSepCV)
		locPredCA.append(locSepCA)
		timePredCA.append(timeSepCA)
		locPredAA.append(locSepAA)
		timePredAA.append(timeSepAA)
	
	dataInp = {'CV_Prediction_Loc': locPredCV, 'CV_Prediction_Time': timePredCV,
			   'CA_Prediction_Loc': locPredCA, 'CA_Prediction_Time': timePredCA,
			   'AA_Prediction_Loc': locPredAA, 'AA_Prediction_Time': timePredAA
			   }
	
	constantVelAndAccel = pd.DataFrame(data=dataInp, index=features.index)
	
	assert constantVelAndAccel['CV_Prediction_Loc'].head().iloc[0] == locPredCV[0]
	assert constantVelAndAccel['CV_Prediction_Time'].head().iloc[0] == timePredCV[0]
	assert constantVelAndAccel['CA_Prediction_Loc'].head().iloc[0] == locPredCA[0]
	assert constantVelAndAccel['CA_Prediction_Time'].head().iloc[0] == timePredCA[0]
	assert constantVelAndAccel['AA_Prediction_Loc'].head().iloc[0] == locPredAA[0]
	assert constantVelAndAccel['AA_Prediction_Time'].head().iloc[0] == timePredAA[0]
	
	return constantVelAndAccel


def predictionConstantVelNextStep(array):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of x and y coordinate of a prediction made with the CV model"""
	
	assert len(array) >= 4  # assume featuresize >= 2
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	
	nextX = array[indexLastX] + v_x
	nextY = array[indexLastY] + v_y
	
	return nextX, nextY


def predictionConstantAccelNextStep(array):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of x and y coordinate of a prediction made with the CA model"""
	
	assert len(array) >= 6  # assume featuresize >= 3
	
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	a_x = v_x - (array[indexLastX - 1] - array[indexLastX - 2])
	a_y = v_y - (array[indexLastY - 1] - array[indexLastY - 2])
	
	t_delta = 1  # 1 time unit between observations
	nextX = array[indexLastX] + t_delta * v_x + 0.5 * t_delta ** 2 * a_x
	nextY = array[indexLastY] + v_y + a_y
	
	return nextX, nextY


def predictionConstantVelSeparator(array, separatorPosition, direction):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of predictions for location and time until intersection with the separator
	made with the CV model"""
	
	assert len(array) >= 4  # assume featuresize >= 2
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	
	xLast = np.transpose(np.array([[array[indexLastX], v_x, array[indexLastY], v_y]]))
	
	A = np.zeros((4,4))
	A[0,1] = 1
	A[2,3] = 1
	
	x_dot = np.matmul(A, xLast)
	
	# x(t) = xPredTo
	
	
	if direction: # moving along x axis:
		locInd = 2
		tarInd = 0
	else:  #moving along y axis
		locInd = 0
		tarInd = 2
	
	assert x_dot[locInd] != 0
	
	deltaT = (separatorPosition - xLast[locInd]) / x_dot[locInd]
	
	intersectionPoint = xLast[tarInd] + deltaT * x_dot[tarInd]
	
	return intersectionPoint[0], deltaT[0]


def predictionAverageAccelSeparator(array, medianAccel, separatorPosition, direction):
	
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	
	a_x = v_x - (array[indexLastX - 1] - array[indexLastX - 2])
	a_y = v_y - (array[indexLastY - 1] - array[indexLastY - 2])

	xLast = np.transpose(np.array([[array[indexLastX], v_x, a_x, array[indexLastY], v_y, a_y]]))
	
	A_h = np.zeros((3,3))
	A_x = A_h.copy()
	A_x[0,1] = 1
	A_x[1,2] = 1
	
	A_0 = np.concatenate((A_x, A_h), axis=0)
	A_1 = np.concatenate((A_h, A_x), axis=0)
	
	A = np.concatenate((A_0, A_1), axis=1)
	
	x_dot = np.matmul(A, xLast)
	
	# print("x vel: {}, x acc: {}".format(v_x, a_x))
	
	if direction: # moving along x axis:
		locInd = 3
		tarInd = 0
	else:  # moving along y axis
		locInd = 0
		tarInd = 3
	
	if medianAccel == 0:
		return predictionConstantVelSeparator(array, separatorPosition, direction)
	else:
		a = 0.5 * medianAccel
		b = x_dot[locInd]
		c = -1 * (separatorPosition - xLast[locInd])
		
		# print("a: {}, b: {}, c: {}".format(a,b,c))
		tempVal = b**2 - 4*a * c
		if tempVal < 0:
			# logging.warning("negative value in sqrt: {}".format(array)) #because negative accelaration along movement axis
			return np.nan, np.nan
		#negative accelaration can lead to not hitting the separator at all
		
		deltaT1 = (-b + tempVal**0.5)/(2*a)
		deltaT2 = (-b - tempVal**0.5)/(2*a)
		
		if deltaT1 <= 0 and deltaT2 <= 0:
			deltaT = np.nan
		else:
			deltaT = min(i for i in [deltaT1, deltaT2] if i > 0)
		
		intersectionPoint = xLast[tarInd] + deltaT * x_dot[tarInd] + 0.5 * deltaT **2 * x_dot[tarInd + 1]
		
		return intersectionPoint[0], deltaT[0]
		
		
def predictionConstantAccelSeparator(array, separatorPosition, direction):
	"""a helper function for the helper function for evaluation.
	given one set of features, returns a tuple of predictions for location and time to intersection
	made with the CA model"""
	
	assert len(array) >= 6  # assume featuresize >= 3
	
	indexLastX = int((len(array)) / 2) - 1  # assuming 2 labels and 2*(featureSize) length)
	indexLastY = int(len(array) - 1)
	
	v_x = array[indexLastX] - array[indexLastX - 1]
	v_y = array[indexLastY] - array[indexLastY - 1]
	a_x = v_x - (array[indexLastX - 1] - array[indexLastX - 2])
	a_y = v_y - (array[indexLastY - 1] - array[indexLastY - 2])
	
	# t_delta = 1  # 1 time unit between observations
	# nextX = array[indexLastX] + t_delta * v_x + 0.5 * t_delta ** 2 * a_x
	# nextY = array[indexLastY] + v_y + a_y

	xLast = np.transpose(np.array([[array[indexLastX], v_x, a_x, array[indexLastY], v_y, a_y]]))
	
	A_h = np.zeros((3,3))
	A_x = A_h.copy()
	A_x[0,1] = 1
	A_x[1,2] = 1
	
	A_0 = np.concatenate((A_x, A_h), axis=0)
	A_1 = np.concatenate((A_h, A_x), axis=0)
	
	A = np.concatenate((A_0, A_1), axis=1)
	
	x_dot = np.matmul(A, xLast)
	
	# print("x vel: {}, x acc: {}".format(v_x, a_x))
	
	if direction: # moving along x axis:
		locInd = 3
		tarInd = 0
	else:  # moving along y axis
		locInd = 0
		tarInd = 3
		
	if x_dot[locInd + 1] == 0:
		return predictionConstantVelSeparator(array, separatorPosition, direction)
	else:

		a = 0.5 * x_dot[locInd + 1]
		b = x_dot[locInd]
		c = -1 * (separatorPosition - xLast[locInd])
		
		# print("a: {}, b: {}, c: {}".format(a,b,c))
		tempVal = b**2 - 4*a * c
		if tempVal < 0:
			# logging.warning("negative value in sqrt: {}".format(array)) #because negative accelaration along movement axis
			return np.nan, np.nan
			#negative accelaration can lead to not hitting the separator at all
			
		deltaT1 = (-b + tempVal**0.5)/(2*a)
		deltaT2 = (-b - tempVal**0.5)/(2*a)
	
		if deltaT1 <= 0 and deltaT2 <= 0:
			deltaT = np.nan
		else:
			deltaT = min(i for i in [deltaT1, deltaT2] if i > 0)
	
		intersectionPoint = xLast[tarInd] + deltaT * x_dot[tarInd] + 0.5 * deltaT **2 * x_dot[tarInd + 1]
		
		return intersectionPoint[0], deltaT[0]
