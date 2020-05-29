import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np


class DataSet:

    def __init__(self):

        self.data = ks.datasets.mnist.load_data()
        self.samples = self.data[1][0]
        self.samples = self.samples.reshape(-1, 28, 28, 1)
        self.n = self.samples.shape[0]
        self.mean = np.mean(self.data[0][0])
        self.std = np.std(self.data[0][0])
        # Samples Normalization
        for i in range(self.n):
            self.samples[i, :, :] = (
                self.samples[i, :, :]-self.mean)/self.std

        self.shape = self.samples.shape
        self.rotations = [20*x for x in range(18)]
        self.m = len(self.rotations)
        self.matrix_transpose = False

    def get_image_shape(self):
        return self.shape

    def get_width(self):
        return self.m

    def get_height(self):
        return self.n

    def get_matrix(self, rows, columns):
        m = []

        if(self.matrix_transpose == False):
            for i in rows:
                x = [tf.keras.preprocessing.image.apply_affine_transform(
                    self.samples[i], theta=self.rotations[j]) for j in columns]
                m.append(x)
        else:
            for i in rows:
                x = [tf.keras.preprocessing.image.apply_affine_transform(
                    self.samples[j], theta=self.rotations[i]) for j in columns]
                m.append(x)

        return tf.convert_to_tensor(m)

    def transpose(self):
        if(self.matrix_transpose == True):
            self.matrix_transpose = False
        else:
            self.matrix_transpose = True
        aux = self.n
        self.n = self.m
        self.m = aux
