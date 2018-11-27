import math
import matplotlib.pyplot as plt
import numpy as np
import locale
import pandas as pd
import os
import random

import sys
sys.path.insert(0, '/home/hornberger/MasterarbeitTobias/implementation')
import loadDataExample as ld
import matplotlib.patches as mpatches
import matplotlib.ticker as tick

def sigmoid(x):
	a = []
	for item in x:
		a.append(1/(1+math.exp(-item)))
	return a


def tanh(x):
	a = []
	for item in x:
		a.append((math.exp(item) - math.exp(-item))/(math.exp(item) + math.exp(-item)))
	return a


def relu(x):
	a = []
	for item in x:
		a.append(np.maximum(0, item))
	return a


def plotFnct():
	x = np.arange(-10., 10., 0.25)
	sig = sigmoid(x)
	# a = plt.plot(x,sig)
	# a = plt.plot(x, tanh(x))
	a = plt.plot(x, relu(x))
	plt.axis([-10, 10, 0, 1])
	plt.xticks(np.arange(-10, 10.001, 5))
	# plt.yticks(np.arange(0, 1.001, 0.25))
	plt.yticks(np.arange(0, 10.001, 2.5))
	# plt.title('Sigmoid')
	plt.title('ReLU')
	plt.grid(True)
	plt.setp(a, linewidth=3, color='r')
	plt.show()


def func(pct, allvals):
	locale.setlocale(locale.LC_ALL, 'de_DE')

	absolute = int(pct/100.*np.sum(allvals))
	return locale.format("%d", absolute, grouping=True)


def plotPieChart():
	labels = 'Kugeln', 'Pfeffer', 'Zylinder', 'Weizen'
	sizes = [7002, 7056, 17049, 8549]
	explode = (0, 0, 0.1, 0.1)

	fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

	wedges, texts, autotexts = ax.pie(sizes, autopct=lambda pct: func(pct, sizes), startangle=90,
									  wedgeprops={"edgecolor": "black", 'linewidth': 2,
												  'antialiased': True})

	# legWedges = wedges
	# for w in legWedges:
	# 	w.set_linewidth(0)

	ax.legend(wedges, labels,
			  title="Schüttgüter",
			  loc="center left",
			  bbox_to_anchor=(1, 0, 0.5, 1))

	plt.setp(autotexts, size=14, weight="bold")

	ax.set_title("Tracks")

	#fig1, ax1 = plt.subplots()
	#ax1.pie(sizes, labels=labels, autopct='%.%',  shadow=True, startangle=90)


	#ax.axis('equal')
	# plt.savefig("pieChart.png", dpi=300)
	plt.show()


def plot( show_axes=False):
	ax = plt.gca()
	# change default range so that new disks will work
	plt.axis('equal')
	ax.set_xlim((-1.5, 1.5))
	ax.set_ylim((-1.5, 1.5))

	if not show_axes:
		ax.set_axis_off()

	disk1 = plt.Circle((0, 0), 1, edgecolor='red', facecolor='green', linestyle='dashed', linewidth=10)
	ax.add_artist(disk1)
	plt.show()
	return

def plotParticleSpeed(inputFile, direction, displayTotal):

	df = pd.read_csv(inputFile)

	numberTracks = (df.shape[1]) / 2

	assert numberTracks == int(numberTracks)

	numberTracks = int(numberTracks)

	print("Number of Tracks: {}".format(numberTracks))

	#plt.axis([0, 20, 80, 100])
	plt.xlabel("Time in Zeitschritt")
	plt.ylabel("speed in pixel/Zeitschritt")

	for i in range(numberTracks):
		trackNo = int(''.join([s for s in df.iloc[:,2*i].name if s.isdigit()]))

		if direction:
			a = df.iloc[:, (2 * i)].values
			b = df.iloc[:, (2 * i + 1)].values
		else:
			a = df.iloc[:, (2 * i + 1)].values
			b = df.iloc[:, (2 * i)].values

		a = ld._removeNans(a)
		b = ld._removeNans(b)

		if np.isnan(a).any() or np.isnan(b).any():
			print("skipping track {}: NaN values in track".format(trackNo))
			continue

		assert len(a) == len(b)

		if len(a) <= 3:
			print("skipping track {}. too few points".format(trackNo))
			continue

		vels = []
		time = [i for i in range(len(b)-1)]

		for i in range(1, len(b)):
			vel = b[i] - b[i-1]
			vels.append(vel)


		plt.title("Track {}".format(os.path.basename(inputFile)))

		if random.random() < 6 * 1/numberTracks:
			features = plt.plot(time, vels, '-o')
			if not displayTotal:
				plt.pause(3)
				f = features.pop(0).remove()

	plt.show()
	plt.close()


def plotImagesPictures():

	N = 2
	numImgKugeln = (51193, 52500)
	numImgPfeffer = (42758, 17500)
	sumKP = [sum(x) for x in zip(numImgKugeln, numImgPfeffer)]
	numImgZyl = (38500, 0)
	sumKPZ = [sum(x) for x in zip(sumKP, numImgZyl)]
	numImgWeiz = (45500, 0)

	width = 0.25
	ind = [0, 1]

	plt.rc('axes', axisbelow=True)

	fig1, ax1 = plt.subplots()
	ax1.grid(True, which='major', axis='y', linestyle='--')
	ax1.set_xlim(-0.5, 1.5)
	# plt.grid.set_axisbelow(True)
	p1 = ax1.bar(ind, numImgKugeln, width, edgecolor='k')
	p2 = ax1.bar(ind, numImgPfeffer, width, edgecolor='k', bottom= numImgKugeln)
	p3 = ax1.bar(ind, numImgZyl, width, bottom=sumKP, edgecolor='k')
	p4 = ax1.bar(ind, numImgWeiz, width, bottom=sumKPZ, edgecolor='k')

	plt.ylabel('Anzahl Bilder')
	ax1.get_yaxis().set_major_formatter(tick.FuncFormatter(lambda  x, p:format(int(x), ',')))
	plt.xticks(ind, ('Band', 'Rutsche'))

	pA = mpatches.Patch(color='C0', label='Kugeln')
	pB = mpatches.Patch(color='C1', label='Pfeffer')
	pC = mpatches.Patch(color='C2', label='Zylinder')
	pD = mpatches.Patch(color='C3', label='Weizen')

	# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Kugeln', 'Pfeffer', 'Zylinder', 'Weizen'), loc=1)
	plt.legend(handles=[pA, pB, pC, pD], loc=1)
	plt.savefig('/home/hornberger/Pictures/ImagesAmount.pdf', bbox_inches="tight", format='pdf', dpi=1200)

	plt.show()

def plotImagesTracks():

	N = 3
	numImgKugeln = (3713, 7712, 5132)
	numImgPfeffer = (0, 7170, 3609)
	sumKP = [sum(x) for x in zip(numImgKugeln, numImgPfeffer)]
	numImgZyl = (4427, 19200, 0)
	sumKPZ = [sum(x) for x in zip(sumKP, numImgZyl)]
	numImgWeiz = (0, 8702, 0)
	sumKPZP = [sum(x) for x in zip(sumKPZ, numImgWeiz)]
	numImgCub = (4357, 0, 0)

	width = 0.25
	ind = [0, 1, 2]

	plt.rc('axes', axisbelow=True)

	fig1, ax1 = plt.subplots()
	ax1.grid(True, which='major', axis='y', linestyle='--')
	ax1.set_xlim(-0.5, 2.5)
	ax1.set_ylim(0, max(sumKPZP) + 1000)
	# plt.grid.set_axisbelow(True)
	p1 = ax1.bar(ind, numImgKugeln, width, edgecolor='k')
	p2 = ax1.bar(ind, numImgPfeffer, width, edgecolor='k', bottom= numImgKugeln)
	p3 = ax1.bar(ind, numImgZyl, width, bottom=sumKP, edgecolor='k')
	p4 = ax1.bar(ind, numImgWeiz, width, bottom=sumKPZ, edgecolor='k')
	p5 = ax1.bar(ind, numImgCub, width, bottom=sumKPZP, edgecolor='k')


	plt.ylabel('Anzahl Tracks')
	ax1.get_yaxis().set_major_formatter(tick.FuncFormatter(lambda  x, p:format(int(x), ',')))
	plt.xticks(ind, ('DEM', 'Band', 'Rutsche'))

	pA = mpatches.Patch(color='C0', label='Kugeln')
	pB = mpatches.Patch(color='C1', label='Pfeffer')
	pC = mpatches.Patch(color='C2', label='Zylinder')
	pD = mpatches.Patch(color='C3', label='Weizen')
	pE = mpatches.Patch(color='C4', label='Plättchen')

	# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Kugeln', 'Pfeffer', 'Zylinder', 'Weizen'), loc=1)
	plt.legend(handles=[pA, pB, pC, pD, pE], loc=1)
	plt.savefig('/home/hornberger/Pictures/ImagesAmountTracks.pdf', bbox_inches="tight", format='pdf', dpi=1200)

	plt.show()

# plotParticleSpeed('/home/hornberger/Projects/MasterarbeitTobias/'
# 				  'data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugeln_004_trackHistory_NothingDeleted.csv', True, True)
#plotPieChart()

# inputFile = '/home/hornberger/Projects/MasterarbeitTobias/data/selbstgesammelteDaten2/Weizen-Band-Juli/weizen_007_trackHistory_NothingDeleted.csv'

# inputFile2 = '/home/hornberger/Projects/MasterarbeitTobias/data/selbstgesammelteDaten2/Zylinder-Band-Juli/zylinder_007_trackHistory_NothingDeleted.csv'

# inputFile3 = '/home/hornberger/Projects/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-BandVorne-Sept/kugeln_rot-002_trackHistory_NothingDeleted.csv'

# plotParticleSpeed(inputFile, True, True)
# plotFnct()

plotImagesTracks()