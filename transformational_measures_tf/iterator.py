import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
from model import Model
from dataset import DataSet
from tqdm import tqdm


class Iterator:

    """
    A class used to iterate over a Model and a DataSet 2D

    ...

    Attributes
    ----------

    Methods
    -------
    get_model()
        Returns the model.
    get_block(width:int,height:int)
        Returns the block iterator.


    """

    def __init__(self, model: Model, dataset: DataSet):
        """
        Parameters
        ----------
        model: Model
        dataset : DataSet
        """

        self.dataset = dataset
        self.model = model
        self.n = self.dataset.get_height()
        self.m = self.dataset.get_width()

    def get_model(self):
        """
        Returns
        -------
        model:Model
            Returns the model

        """
        return self.model

    def get_block(self, height: int, width: int):
        """

        Parameters
        ----------
        height:int
        widht:int

        Returns
        -------
        x:Iterator
            Returns a "vertical" iterator where each item in this
            iterator is a tuple (height,blocks_iterator) where 
            blocks_iterator is an "horizontal" iterator in which 
            each item is a tuple (width,block_activations) where 
            block_activations is an object that indexes a three 
            dimensional list through a tuple (i,j,l) where i 
            represents the row and j represents the column 
            corresponding to the block and l represents the l-layer
            of a model. The values in the list are the activations 
            values in the layer l for the input located in the row i 
            and the column j, and width is the width of the block and 
            the height of the block is the height returned previously.
        """
        batchs = self.generate_batchs(height, self.n)
        for batch in tqdm(batchs):
            yield len(batch), self.get_block_activations(batch, width)

    def get_block_activations(self, batch_vertical, width):
        batchs = self.generate_batchs(width, self.m)
        for batch in batchs:
            x = self.dataset.get_matrix(batch_vertical, batch)
            bloque = self.model.predict(x)
            bloque.set_rows_dataset(batch_vertical)
            bloque.set_columns_dataset(batch)
            bloque.set_state_transpose_dataset(self.dataset.is_transpose())
            yield len(batch), bloque

    def generate_batchs(self, size_batch, n):
        aux = range(n)
        h = int(n/size_batch)
        batchs = [aux[j*size_batch:(j+1)*size_batch] for j in range(h)]
        if(n % size_batch > 0):
            batchs.append(aux[h*size_batch:n])
            h = h+1
        return batchs
