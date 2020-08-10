import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np


class Model:

    """
    A class used to contain a Keras Model

    ...

    Attributes
    ----------

    Methods
    -------
    predict(tensor)
        Predict the matrix of inputs given by the tensor
    get_model_structure()
        Returns an instance of ModelStructure with the data of the model structure.
    get_name_layer(k:int)
        Returns the name of the layer k.
    evaluate(x,y)
        Returns a list with the loss and the accuracy for the tuple (x,y)
    """

    def __init__(self, model_keras: tf.keras.Model, model_path: str):
        """
        Parameters
        ----------
        model_keras : tf.keras.Model
            a keras model
        model_path : str
            a path to load a keras model
        """

        if(model_keras != None):
            self.model_keras = model_keras
        else:
            self.model_keras = tf.keras.models.load_model(model_path)

        self.layers_outpus = [
            layer.output for layer in self.model_keras.layers]
        self.layers_names = [layer.name for layer in self.layers_outpus]
        self.layers_names = [
            name[:name.index('/')]for name in self.layers_names]

        self.model_layers = ks.models.Model(
            inputs=self.model_keras.input, outputs=self.layers_outpus)

        self.model_structure = ModelStructure(self.model_keras)

    def predict(self, tensor: tf.Tensor):
        """Predict a matrix of inputs given by the tensor

        Parameters
        ----------
        tensor:tf.Tensor

        Returns
        -------
        x:X
            Returns an object that indexes a three dimensional list
            through a tuple (i,j,l) where i represents the row and j
            represents the column corresponding to the input matrix
            and l represents the l-layer of a model. The values in the
            list are the activations values in the layer l for the
            input located in the row i and the column j.
        """

        aux = []
        aux.append(tensor.shape[0]*tensor.shape[1])
        for i in range(len(tensor.shape)-2):
            aux.append(tensor.shape[i+2])

        x = tf.reshape(tensor, aux)

        x = self.model_layers.predict(x)

        return Block(x, tensor.shape[0], tensor.shape[1], self.model_structure)

    def get_model_structure(self):
        """

        Parameters
        ----------

        Returns
        -------
        x:ModelStructure

        """
        return self.model_structure

    def get_name_layer(self, k):
        """

        Parameters
        ----------
        k:int
            the index of the layer
        Returns
        -------
        x:str
            Returns the name of the layer k.

        """
        return self.layers_names[k]

    def evaluate(self, x, y):
        """

        Parameters
        ----------
        x
            the input of the model
        y
            the desired output for x


        Returns
        -------
        result:list
            Returns a list with the loss and the accuracy for the tuple (x,y)

        """
        return self.model_keras.evaluate(x=x, y=y)


class ModelStructure:
    """
    A class used to represent the model structure
    ...

    Attributes
    ----------
    layers:list
        a list with the layer structure for each layer
    number_layers:int
        the number of layers in the model
    map_layers_features:list
        a list containing a map between the layers and the layers with features
    layers_features:list
        a list with the layer structure for each layer with features
    number_layers_features:int
        the number of layers with features
    Methods
    -------

    """

    def __init__(self, model: tf.keras.Model = None):
        """
        Parameters
        ----------
        model : tf.keras.Model
            a keras model
        """
        self.layers = []
        self.number_layers = 0
        self.map_layers_features = []
        self.layers_features = []
        self.number_layers_features = 0

        if model != None:

            self.layers_outpus = [layer.output for layer in model.layers]
            k = 0
            self.number_layers_features = 0
            for layer in self.layers_outpus:

                shape = layer.shape[1:]
                layer_structure = LayerStructure()
                layer_structure.shape = shape
                h = 1
                for i in shape:
                    h = h*i
                layer_structure.number_activations = h

                if len(shape) > 2:
                    layer_structure.number_features = shape[-1]
                    layer_structure.shape_feature = shape[0:-1]
                    self.layers_features.append(layer_structure)
                    self.map_layers_features.append(k)
                    self.number_layers_features += 1
                else:
                    layer_structure.number_features = 0
                    pass

                self.layers.append(layer_structure)
                k = k+1

            self.number_layers = k


class LayerStructure:
    """
    A class used to represent the structure of a layer in a keras model 
    ...

    Attributes
    ----------
    number_features:int
        the number of features in the layer
    shape_feature:tuple
        the shape for the features in the layer
    number_activations
        the number of activations in the layer
    shape:tuple
        the shape of the layer

    Methods
    -------

    """

    def __init__(self):
        self.number_features = None
        self.shape_feature = None
        self.number_activations = None
        self.shape = None


class Block:
    """
    A class used to represent the output when the model predicts a block of inputs. The block can be indexed 
    by three or four indexes to get a particular output. Besides, the block has some information of the 
    dataset from which the inputs come. 
    ...

    Attributes
    ----------
    height:int
        the height of the block
    width:int
        the width of the block
    model_structure:ModelStructure
        the model structure

    Methods
    -------
    set_rows_dataset(rows:list)

    set_columns_dataset(columns:list)

    set_state_transpose_dataset(state:bool)

    get_rows_dataset()

    get_columns_dataset()

    is_dataset_transpose()

    """

    def __init__(self, x, height: int, width: int, model_structure: ModelStructure):
        self.x = x
        self.height = height
        self.width = width
        self.model_structure = model_structure
        self.rows_dataset = None
        self.columns_dataset = None
        self.state_transpose_dataset = None

    def __getitem__(self, tuple):
        # tuple[0]=i,tuple[1]=j,tuple[2]=l,tuple[3]=f

        if len(tuple) == 3:
            layer = self.x[tuple[2]][tuple[0]*self.width+tuple[1]]
            return layer

        if len(tuple) == 4:
            k = self.model_structure.map_layers_features[tuple[2]]
            layer = self.x[k][tuple[0]*self.width+tuple[1]]
            feature = tf.unstack(layer, axis=tf.rank(layer)-1)
            feature = feature[tuple[3]]
            return feature

        return None

    def __setitem__(self, tuple, value):

        if len(tuple) == 3:
            self.x[tuple[2]][tuple[0]*self.width+tuple[1]] = value

        if len(tuple) == 4:
            layer = self.x[tuple[2]][tuple[0]*self.width+tuple[1]]
            axis = tf.rank(layer)-1
            layer_aux = tf.unstack(layer, axis=axis)
            layer_aux[tuple[3]] = value
            self.x[tuple[2]][tuple[0]*self.width+tuple[1]
                             ] = tf.stack(layer_aux, axis=axis)

    def set_rows_dataset(self, rows: list):
        self.rows_dataset = rows

    def set_columns_dataset(self, columns: list):
        self.columns_dataset = columns

    def set_state_transpose_dataset(self, state: bool):
        self.state_transpose_dataset = state

    def get_rows_dataset(self):
        return self.rows_dataset

    def get_columns_dataset(self):
        return self.columns_dataset

    def is_dataset_transpose(self):
        return self.state_transpose_dataset
