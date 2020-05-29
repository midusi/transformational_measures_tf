import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np


class Model:

    def __init__(self, path_name):

        self.original_model = tf.keras.models.load_model(path_name)
        self.name_model = path_name
        # a new layer model is created for accessing to the activations

        self.layers_outpus = [
            layer.output for layer in self.original_model.layers]
        self.layers_names = [layer.name for layer in self.layers_outpus]
        self.layers_names = [name[:name.index('/')]
                             for name in self.layers_names]

        self.model_layers = ks.models.Model(
            inputs=self.original_model.input, outputs=self.layers_outpus)

        self.nactivations = 0  # total number of activations

        self.layers_nactivations = []  # list of number of activations for layers

        for layer in self.layers_outpus:
            a = 1
            for i in layer.shape:
                if(i != None):
                    a = a*i

            self.nactivations = self.nactivations+a
            self.layers_nactivations.append(a)

        # variable que almacena el numero de layers del modelo
        self.nlayers = len(self.layers_outpus)

    def predict(self, tensor):

        aux = []
        aux.append(tensor.shape[0]*tensor.shape[1])
        for i in range(len(tensor.shape)-2):
            aux.append(tensor.shape[i+2])

        x = tf.reshape(tensor, aux)
        #x = tf.reshape(tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]))

        x = self.model_layers.predict(x)

        return X(x, tensor.shape[1])

    def size_layers(self):
        return self.nlayers

    def size_activations(self):
        return self.nactivations

    def size_activations_layers(self):
        return self.layers_nactivations


class X:
    def __init__(self, x, n):
        self.x = x
        self.n = n

    def __getitem__(self, tuple):
        # tuple[0]=i,tuple[1]=j,tuple[2]=l
        layer = self.x[tuple[2]][tuple[0]*self.n+tuple[1]]
        layer = tf.reshape(layer, [-1])
        return layer
