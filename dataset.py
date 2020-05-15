import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np


class Dataset:

    def __init__(self):

        self.datos = ks.datasets.mnist.load_data()
        self.muestras = self.datos[1][0]
        self.muestras = self.muestras.reshape(-1, 28, 28, 1)
        self.n = self.muestras.shape[0]
        self.media = np.mean(self.datos[0][0])
        self.var = np.std(self.datos[0][0])
        # Normalizod_in las variables de entrada
        for i in range(self.n):
            self.muestras[i, :, :] = (
                self.muestras[i, :, :]-self.media)/self.var

        self.shape = self.muestras.shape
        self.rotaciones = [20*x for x in range(18)]
        self.m = len(self.rotaciones)

    def get_image_shape(self):
        return self.shape

    def get_ancho(self):
        return self.m

    def get_alto(self):
        return self.n

    def get_matriz(self, filas, columnas):
        m = []
        for i in filas:
            x = [tf.keras.preprocessing.image.apply_affine_transform(
                self.muestras[i], theta=self.rotaciones[j]) for j in columnas]
            m.append(x)
        return tf.convert_to_tensor(m)

