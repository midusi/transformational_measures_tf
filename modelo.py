import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np


class Modelo:

    def __init__(self, ruta_nombre):

        self.modelo_original = tf.keras.models.load_model(ruta_nombre)
        self.nombre_modelo = ruta_nombre
        # se crea un nuevo modelo en capas para poder acceder a las activaciones

        self.capas_salidas = [
            capa.output for capa in self.modelo_original.layers]
        self.capas_nombres = [capa.name for capa in self.capas_salidas]
        self.capas_nombres = [nombre[:nombre.index(
            '/')] for nombre in self.capas_nombres]

        self.modelo_capas = ks.models.Model(
            inputs=self.modelo_original.input, outputs=self.capas_salidas)

        self.nactivaciones = 0  # variable que almacena el numero de activaciones totales
        # lista que almacena el numero de activaciones de cada capa
        self.capas_nactivaciones = []

        for capa in self.capas_salidas:
            a = 1
            for i in capa.shape:
                if(i != None):
                    a = a*i

            self.nactivaciones = self.nactivaciones+a
            self.capas_nactivaciones.append(a)

        # variable que almacena el numero de capas del modelo
        self.ncapas = len(self.capas_salidas)

    def predecir(self, tensor):
        x = tf.reshape(
            tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]))
        x = self.modelo_capas.predict(x)

        return X(x, tensor.shape[1])

    def numero_capas(self):
        return self.ncapas

    def numero_activaciones(self):
        return self.nactivaciones

    def numero_activaciones_por_capa(self):
        return self.capas_nactivaciones


class X:
    def __init__(self, x, n):
        self.x = x
        self.n = n

    def __getitem__(self, tupla):
        # tupla[0]=i,tupla[1]=j,tupla[2]=l
        capa = self.x[tupla[2]][tupla[0]*self.n+tupla[1]]
        capa = tf.reshape(capa, [-1])
        return capa
