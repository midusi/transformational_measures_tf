from abc import ABC, abstractmethod
import tensorflow as tf


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
