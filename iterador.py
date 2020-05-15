import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
from modelo import Modelo
from dataset import Dataset


class Iterador:

    def __init__(self, modelo: Modelo, dataset: Dataset):
        self.dataset = dataset
        self.modelo = modelo
        self.n = self.dataset.get_alto()
        self.n = 200
        self.m = self.dataset.get_ancho()

    def por_bloques_muestras_primero(self, alto, ancho):
        batchs = self.generar_batchs(alto, self.n)
        i = -1
        for batch in batchs:
            i += 1
            yield (i/len(batchs)), len(batch), self.get_bloques_activaciones_horizontal(batch, ancho)

    def get_bloques_activaciones_horizontal(self, batch_muestras, ancho):
        batchs = self.generar_batchs(ancho, self.m)
        for batch in batchs:
            x = self.dataset.get_matriz(batch_muestras, batch)
            x = self.modelo.predecir(x)
            yield len(batch), x

    def por_bloques_transformaciones_primero(self, alto, ancho):
        batchs = self.generar_batchs(alto, self.m)
        i = -1
        for batch in batchs:
            i += 1
            yield (i/len(batchs)), len(batch), self.get_bloques_activaciones_vertical(batch, ancho)

    def get_bloques_activaciones_vertical(self, batch_transformaciones, ancho):
        batchs = self.generar_batchs(ancho, self.n)
        for batch in batchs:
            x = self.dataset.get_matriz(batch, batch_transformaciones)
            x = self.modelo.predecir(x)
            yield len(batch), x

    def generar_batchs(self, tamaño_batch, n):
        aux = range(n)
        h = int(n/tamaño_batch)
        batchs = [aux[j*tamaño_batch:(j+1)*tamaño_batch] for j in range(h)]
        if(n % tamaño_batch > 0):
            batchs.append(aux[h*tamaño_batch:n])
            h = h+1
        return batchs
