#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import logging
import sys
import time
from collections import OrderedDict

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from adjustText import adjust_text


def redrawNext(pandasLost, imageLoc, units):

	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
	relevantColumns = ['NNpixelErrorX', 'NNpixelErrorY', 'NNpixelErrorTotal', 'CVpixelErrorX', 'CVpixelErrorY', 'CVpixelErrorTotal', 'CApixelErrorX', 'CApixelErrorY', 'CApixelErrorTotal']

	reducedRelColumns = [relevantColumns[2], relevantColumns[5], relevantColumns[8]]
	reducedRelColumnsLabel = ['Fehler NN', 'Fehler CV', 'Fehler CA']

	#_printPDfull(pandasLost[reducedRelColumns].describe())

	for i in reducedRelColumns:
		pandasLost.loc[:, i] *= 1000


	logging.info("number of predictions with error > 3: {}".format((pandasLost['NNpixelErrorTotal'] > 3).sum()))

	# TODO: Maybe save column total Pixelerror of current prediction so it can be compared to other schüttgüter

	plt.rc('grid', linestyle=":")
	fig1, ax1 = plt.subplots()
	ax1.boxplot([pandasLost[i] for i in reducedRelColumns], showfliers=False)
	ax1.yaxis.grid(True)
	xtickNames = plt.setp(ax1, xticklabels=reducedRelColumnsLabel)
	plt.setp(xtickNames, rotation=45, fontsize=8)
	#ax1.set_title('Boxplot Location Error')
	ax1.set_ylabel('Fehler in {}'.format(units['loc']))  # a little bit hacky

	fig1.tight_layout()

	# plt.show()
	fig2, ax2 = plt.subplots()
	ax2.hist([pandasLost[i] for i in reducedRelColumns],
			 bins=40, label=reducedRelColumns)
	# # plt.yscale('log')
	# ax.style.use('seaborn-muted')
	#ax2.set_title('Error Histogram')
	ax2.set_ylabel('Anzahl Elemente')
	ax2.set_xlabel('Fehler in {}'.format(units['loc']))
	plt.legend(loc=1)
	fig1.savefig(imageLoc + '/evaluation_NextStep_LocationError_' + time_stamp + '.pdf', format='pdf', dpi=1200)
	fig2.savefig(imageLoc + '/evaluation_NextStep_ErrorHistogram_' + time_stamp + '.pdf', format='pdf', dpi=1200)


	logging.info("Saving dataframe:")
	pandasLost.to_pickle(imageLoc + '/pandasLostDataframe' + time_stamp + '.pkl')
	plt.show()


def redraw(pandasLost, imageLoc, units):

	time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

	relevantColumnsLoc = ['NNpixelErrorPosBalken', 'CVpixelErrorPosBalken', 'CVBCpixelErrorPosBalken', 'CApixelErrorPosBalken']
	relevantColumnsLocLabel = ['Fehler NN', 'Fehler CV', 'Fehler CVBC', 'Fehler CA']
	relevantColumnsTime = ['NNerrorTime', 'CVerrorTime', 'CVBCerrorTime', 'CAerrorTime', 'AAerrorTime', 'IAerrorTime']
	relevantColumnsTimeLabel = ['Fehler NN', 'Fehler CV', 'Fehler CVBC', 'Fehler CA', 'Fehler AA', 'Fehler IA']
	# logging.info("\n{}".format(pandasLost[relevantColumns]))

	# for i in relevantColumnsLoc:
	# 	pandasLost.loc[:, i] *= 1000

	# for i in relevantColumnsTime:
	# 	pandasLost.loc[:, i] *= 100

	# logging.info("number of predictions with error > 3: {}".format((pandasLost['NNpixelErrorTotal'] > 3).sum()))
	plt.rc('grid', linestyle=":")
	fig1, ax1 = plt.subplots()
	# ax1.grid(True)
	ax1.yaxis.grid(True)
	ax1.boxplot([pandasLost[i] for i in relevantColumnsLoc], showfliers=False, labels=relevantColumnsLocLabel)
	xtickNames = plt.setp(ax1, xticklabels=relevantColumnsLocLabel)
	plt.setp(xtickNames, rotation=45, fontsize=8)
	#ax1.set_title('Boxplot Location Error')
	ax1.set_ylabel('Räumlicher Fehler in {}'.format(units['loc']))  # a little bit hacky
	fig1.tight_layout()

	# plt.show()
	fig2, ax2 = plt.subplots()
	ax2.hist([pandasLost[i] for i in relevantColumnsLoc],
			 bins=40, label=relevantColumnsLoc)
	#ax2.set_title('Histogram Location Error')
	ax2.set_ylabel('Anzahl Elemente')  # a little bit hacky
	ax2.set_xlabel('Räumlicher Fehler in {}'.format(units['loc']))  # a little bit hacky
	ax2.legend(loc=1)
	# # plt.yscale('log')
	# ax.style.use('seaborn-muted')

	fig3, ax3 = plt.subplots()
	# ax3.grid(True)
	ax3.yaxis.grid(True)
	ax3.boxplot([pandasLost[i] for i in relevantColumnsTime], showfliers=False)
	xtickNamesAx3 = plt.setp(ax3, xticklabels=relevantColumnsTimeLabel)
	plt.setp(xtickNamesAx3, rotation=45, fontsize=8)
	#ax3.set_title('Boxplot Time Error')
	ax3.set_ylabel('Zeitlicher Fehler in {}'.format(units['time']))  # a little bit hacky
	fig3.tight_layout()

	fig4, ax4 = plt.subplots()
	ax4.hist([pandasLost[i] for i in relevantColumnsTime],
			 bins=40, label=relevantColumnsTime)
	#ax4.set_title('Histogram Time Error')
	ax4.set_ylabel('Anzahl Elemente')  # a little bit hacky
	ax4.set_xlabel('Zeitlicher Fehler in {}'.format(units['time']))  # a little bit hacky
	ax4.legend(loc=1)
	plt.tight_layout()
	logging.info("Saving evaluation images to {}".format(imageLoc))
	fig1.savefig(imageLoc + '/evaluation_Separator_LocationErrorBoxplot_' + time_stamp + '.pdf', format='pdf', dpi=1200)
	fig2.savefig(imageLoc + '/evaluation_Separator_LocationErrorHistogram_' + time_stamp + '.pdf', format='pdf', dpi=1200)
	fig3.savefig(imageLoc + '/evaluation_Separator_TimeErrorBoxplot_' + time_stamp + '.pdf', format='pdf', dpi=1200)
	fig4.savefig(imageLoc + '/evaluation_Separator_TimeErrorHistogram_' + time_stamp + '.pdf', format='pdf', dpi=1200)

	#logging.info("Saving dataframe:")
	#pandasLost.to_pickle(imageLoc + '/pandasLostDataframe' + time_stamp + '.pkl')
	plt.show()



logging.basicConfig(level=logging.INFO)
logging.info('test start')


with open("/home/hornberger/Projects/MasterarbeitTobias/images/Simulated-NextStep-45kDecay-cuboid/pandasLostDataframe2018-12-10_19.08.06.pkl", 'rb') as f:
	df = pickle.load(f)
units = {}
units['loc'] = 'mm'
units['time'] = '1/100 Frames'

imageLoc = "/home/hornberger/Desktop/test"

redrawNext(df, imageLoc, units)


