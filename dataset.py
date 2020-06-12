
from abc import ABC, abstractmethod 



class DataSet(ABC):

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
