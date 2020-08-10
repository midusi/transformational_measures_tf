
from abc import ABC, abstractmethod


class DataSet(ABC):
    """
    A class used to represent a two dimensional dataset

    ...

    Attributes
    ----------

    Methods
    -------
    get_data_shape()
        Returns the shape of the data in the matrix.
    get_width()
        Returns the matrix width.
    get_height()
        Returns the matrix height.
    get_matrix(row:int,column:int)
        Returns a submatrix conformed by the rows and columns passed as arguments.
    transpose()
        Transpose the DataSet

    """
    # abstract method

    def get_data_shape(self):
        pass
    # abstract method

    def get_width(self):
        pass
    # abstract method

    def get_height(self):
        pass
    # abstract method

    def get_matrix(self, rows, columns):
        pass
    # abstract method

    def transpose(self):
        pass

    # abstract method
    def is_transpose(self):
        pass

    # abstract method
    def get_testing_data(self):
        pass
