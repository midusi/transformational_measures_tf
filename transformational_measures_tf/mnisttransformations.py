import random
import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from transformationsdataset import TransformationsDataSet


class Transformation(ABC):
    """
    A class used to contain a Keras Model

    ...

    Attributes
    ----------

    Methods
    -------
    predict(tensor)
        Predict the matrix of inputs given by the tensor

    """

    # abstract method
    def apply(self, tensor):
        pass

    # abstract method
    def apply_inverse(self, tensor):
        pass


class AffineTransformation(Transformation):

    def __init__(self, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1):

        self.data = ks.datasets.mnist.load_data()
        self.theta = theta
        self.tx = tx
        self.ty = ty
        self.shear = shear
        self.zx = zx
        self.zy = zy

    # overriding abstract method
    def apply(self, tensor):
        return tf.keras.preprocessing.image.apply_affine_transform(tensor, theta=self.theta, tx=self.tx, ty=self.ty, shear=self.shear, zx=self.zx, zy=self.zy)

    # overriding abstract method
    def apply_inverse(self, tensor):
        return tf.keras.preprocessing.image.apply_affine_transform(tensor, theta=-self.theta, tx=-self.tx, ty=-self.ty, shear=-self.shear, zx=-self.zx, zy=-self.zy)


class MnistTransformationsDataSet(TransformationsDataSet):

    def __init__(self):

        self.data = ks.datasets.mnist.load_data()
        self.results = self.data[1][1]
        self.samples = self.data[1][0]
        self.samples = self.samples.reshape(-1, 28, 28, 1)
        self.height = self.samples.shape[0]
        self.mean = np.mean(self.data[0][0])
        self.std = np.std(self.data[0][0])
        # Samples Normalization
        for i in range(self.height):
            self.samples[i, :, :] = (self.samples[i, :, :]-self.mean)/self.std

        self.original_samples = self.samples
        self.original_results = self.results

        self.shape = self.samples.shape

        self.transformations = [AffineTransformation()]
        self.width = len(self.transformations)
        self.matrix_transpose = False
        self.size_transformations_groups = 7

    # overriding abstract method
    def get_data_shape(self):
        return self.shape

    # overriding abstract method
    def get_width(self):
        return self.width

    # overriding abstract method
    def get_height(self):
        return self.height

    # overriding abstract method
    def get_matrix(self, rows, columns):
        m = []

        if(self.matrix_transpose == False):
            for i in rows:
                x = [self.transformations[j].apply(tf.make_ndarray(
                    tf.make_tensor_proto(self.samples[i]))) for j in columns]
                m.append(x)
        else:
            for i in rows:
                x = [self.transformations[i].apply(tf.make_ndarray(
                    tf.make_tensor_proto(self.samples[j]))) for j in columns]
                m.append(x)

        return tf.convert_to_tensor(m)

    # overriding abstract method
    def transpose(self):
        if(self.matrix_transpose == True):
            self.matrix_transpose = False
        else:
            self.matrix_transpose = True
        aux = self.height
        self.height = self.width
        self.width = aux

    # overriding abstract method
    def apply_inverse_transformation(self, tensor: tf.Tensor, index_transformation):
        if len(tensor.shape) == 2:
            shape_aux = tensor.shape
            shape = (tensor.shape[0], tensor.shape[1], 1)
            tensor = tf.reshape(tensor, shape)
            array = tf.make_ndarray(tf.make_tensor_proto(tensor))
            img = self.transformations[index_transformation].apply_inverse(
                array)
            img = tf.reshape(img, shape_aux)
            return img
        else:
            return tensor

    # overriding abstract method
    def get_testing_data(self):

        x = []
        y = []

        for i in range(self.samples.shape[0]):
            for t in self.transformations:
                x.append(t.apply(tf.make_ndarray(
                    tf.make_tensor_proto(self.samples[i]))))
                y.append(self.results[i])

        x = tf.stack(x)
        y = tf.stack(y)
        return (x, y)

    def select_random_samples(self, n: int):
        indexes = random.sample(range(self.original_samples.shape[0]), n)
        self.select_samples(indexes)

    def select_samples(self, indexes):
        aux_list = []
        self.results = []
        n = 0
        for i in indexes:
            aux_list.append(self.original_samples[i, :, :, :])
            self.results.append(self.original_results[i])
            n = n+1
        aux_list = tf.stack(aux_list, axis=0)
        self.samples = aux_list

        if self.matrix_transpose == False:
            self.height = n
        else:
            self.width = n

    # overriding abstract method
    def is_transpose(self):
        return self.matrix_transpose

    # overriding abstract method
    def get_size_transformations_groups(self):
        return self.size_transformations_groups

    # overriding abstract method
    def set_transformation_group(self, group, n):

        self.transformations = []
        if group == 0:
            for i in range(n):
                self.transformations.append(AffineTransformation())
        if group == 1:
            h = int(360/n)
            for i in range(n):
                self.transformations.append(AffineTransformation(theta=i*h))
        elif group == 2:
            width = self.original_samples[0].shape[1]
            if n <= width:
                h = int(width/n)
                for i in range(n):
                    self.transformations.append(AffineTransformation(tx=i*h))
            else:
                for i in range(n):
                    self.transformations.append(AffineTransformation(tx=i))
        elif group == 3:
            height = self.original_samples[0].shape[0]
            if n <= height:
                h = int(height/n)
                for i in range(n):
                    self.transformations.append(AffineTransformation(ty=i*h))
            else:
                for i in range(n):
                    self.transformations.append(AffineTransformation(ty=i))
        elif group == 4:
            n1 = int(n/2)
            n2 = n1
            if n != (n1+n2):
                n1 = n1+1

            height = self.original_samples[0].shape[0]
            if n1 <= height:
                h = int(height/n1)
                for i in range(n1):
                    self.transformations.append(AffineTransformation(ty=i*h))
            else:
                for i in range(n1):
                    self.transformations.append(AffineTransformation(ty=i))

            width = self.original_samples[0].shape[1]
            if n2 <= width:
                h = int(width/n2)
                for i in range(n2):
                    self.transformations.append(AffineTransformation(tx=i*h))
            else:
                for i in range(n2):
                    self.transformations.append(AffineTransformation(tx=i))

        elif group == 5:
            n1 = int(n/2)
            n2 = n1
            if n != (n1+n2):
                n1 = n1+1

            h = int(360/n1)
            for i in range(n1):
                self.transformations.append(AffineTransformation(theta=i*h))

            height = self.original_samples[0].shape[0]
            if n2 <= height:
                h = int(height/n2)
                for i in range(n2):
                    self.transformations.append(AffineTransformation(ty=i*h))
            else:
                for i in range(n2):
                    self.transformations.append(AffineTransformation(ty=i))

        elif group == 6:
            n1 = int(n/2)
            n2 = n1
            if n != (n1+n2):
                n1 = n1+1

            h = int(360/n1)
            for i in range(n1):
                self.transformations.append(AffineTransformation(theta=i*h))

            width = self.original_samples[0].shape[1]
            if n2 <= width:
                h = int(width/n2)
                for i in range(n2):
                    self.transformations.append(AffineTransformation(tx=i*h))
            else:
                for i in range(n2):
                    self.transformations.append(AffineTransformation(tx=i))

        else:
            pass

        self.transformation_group = group

        if self.matrix_transpose == False:
            self.width = n
        else:
            self.height = n
