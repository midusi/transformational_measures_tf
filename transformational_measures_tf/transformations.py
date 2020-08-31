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
    def apply(self, tensor_image: tf.Tensor):

        if tf.rank(tensor_image) == 2:
            shape_aux = tensor_image.shape
            shape = (tensor_image.shape[0], tensor_image.shape[1], 1)
            image = tf.reshape(tensor_image, shape)
            image = tf.make_ndarray(tf.make_tensor_proto(image))
            image = tf.keras.preprocessing.image.apply_affine_transform(
                image, theta=self.theta, tx=self.tx, ty=self.ty, shear=self.shear, zx=self.zx, zy=self.zy)
            image = tf.reshape(image, shape_aux)
            return image

        if tf.rank(tensor_image) == 3:
            image = tf.make_ndarray(tf.make_tensor_proto(tensor_image))
            return tf.keras.preprocessing.image.apply_affine_transform(image, theta=self.theta, tx=self.tx, ty=self.ty, shear=self.shear, zx=self.zx, zy=self.zy)

    # overriding abstract method
    def apply_inverse(self,  tensor_image: tf.Tensor):

        if tf.rank(tensor_image) == 2:
            shape_aux = tensor_image.shape
            shape = (tensor_image.shape[0], tensor_image.shape[1], 1)
            image = tf.reshape(tensor_image, shape)
            image = tf.make_ndarray(tf.make_tensor_proto(image))
            image = tf.keras.preprocessing.image.apply_affine_transform(
                image, theta=-self.theta, tx=-self.tx, ty=-self.ty, shear=-self.shear, zx=-self.zx, zy=-self.zy)
            image = tf.reshape(image, shape_aux)
            return image

        if tf.rank(tensor_image) == 3:
            image = tf.make_ndarray(tf.make_tensor_proto(tensor_image))
            return tf.keras.preprocessing.image.apply_affine_transform(image, theta=-self.theta, tx=-self.tx, ty=-self.ty, shear=-self.shear, zx=-self.zx, zy=-self.zy)
