import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from model import Model
from mnistdataset import MnistDataSet
from iterator import Iterator
from variance import Variance

path = ""


model = Model(path+"/modelosimple.h5")
dataset = MnistDataSet()
iterator = Iterator(model, dataset)

variance = Variance(iterator)

variance.compute(10, 2)

plt.figure()
plt.title("Transformational Variance")
plt.plot(model.layers_names, variance.variance_layers)
print(variance.variance_layers)

model = Model(path+"/modelosimple.h5")
dataset = DataSet()
dataset.transpose()
iterator = Iterator(model, dataset)

variance = Variance(iterator)

variance.compute(10, 2)

plt.figure()
plt.title("Muestral Variance")
plt.plot(model.layers_names, variance.variance_layers)
print(variance.variance_layers)
