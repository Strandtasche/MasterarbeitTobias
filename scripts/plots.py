import math
import matplotlib.pyplot as plt
import numpy as np

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