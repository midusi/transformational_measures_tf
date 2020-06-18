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
    size_layers()
        Returns the amount of layers in the model.
    size_activations()
        Returns the amount of activations in the model.
    size_activations_layers()
        Returns a list with the amount of activations in each layer of the model.
    
    """


    def __init__(self,model_keras:tf.keras.Model,model_path:str):
        
        """
        Parameters
        ----------
        model_keras : tf.keras.Model
            a keras model
        model_path : str
            a path to load a keras model
        """

        if(model_keras!=None):
            self.model_keras = model_keras
        else:
            self.model_keras = tf.keras.models.load_model(model_path)

        self.layers_outpus = [
            layer.output for layer in self.model_keras.layers]
        self.layers_names = [layer.name for layer in self.layers_outpus]
        self.layers_names = [name[:name.index('/')]
                             for name in self.layers_names]

        self.model_layers = ks.models.Model(
            inputs=self.model_keras.input, outputs=self.layers_outpus)

        self.nactivations = 0  # total number of activations

        self.layers_nactivations = []  # list of number of activations for layers

        for layer in self.layers_outpus:
            a = 1
            for i in layer.shape:
                if(i != None):
                    a = a*i
            self.nactivations = self.nactivations+a
            self.layers_nactivations.append(a)

        # variable que almacena el numero de layers del modelo
        self.nlayers = len(self.layers_outpus)

    def predict(self, tensor:tf.Tensor):
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

        return X(x, tensor.shape[1])

    def size_layers(self):
        """
        
        Parameters
        ----------   
        
        Returns
        -------
        x:int
            Returns the amount of layers in the model.    

        """
        return self.nlayers

    def size_activations(self):
        """
        
        Parameters
        ----------   
        
        Returns
        -------
        x:int
            Returns the amount of activations in the model.    

        """
        return self.nactivations

    def size_activations_layers(self):
        """
        
        Parameters
        ----------   
        
        Returns
        -------
        x:int
            Returns a list with the amount of activations in each layer of the model.    

        """
        return self.layers_nactivations


class X:
    """
    A class used to convert a two dimensional list into a three dimensional list

    ...

    Attributes
    ----------
    
    Methods
    -------
    
    """
    
    def __init__(self, x, n):
        self.x = x
        self.n = n

    def __getitem__(self, tuple):
        # tuple[0]=i,tuple[1]=j,tuple[2]=l
        layer = self.x[tuple[2]][tuple[0]*self.n+tuple[1]]
        layer = tf.reshape(layer, [-1])
        return layer
