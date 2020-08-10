
from abc import ABC, abstractmethod
import tensorflow as tf
from dataset import DataSet


class TransformationsDataSet(DataSet):
    """
    A class used to have an abstract representation of a two dimensional transformations dataset

    ...

    Attributes
    ----------

    Methods
    -------


    """

    # abstract method
    def apply_inverse_transformation(self, tensor: tf.Tensor, index_transformation: int):
        pass

    # abstract method
    def get_size_transformations_groups(self):
        pass

    # abstract method
    def set_transformation_group(self, group: int, n: int):
        pass
