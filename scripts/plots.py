import math
import matplotlib.pyplot as plt
import numpy as np
import locale


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
	
plotPieChart()
# plotFnct()