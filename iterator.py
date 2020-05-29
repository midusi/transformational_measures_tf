import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
from model import Model
from dataset import DataSet
from tqdm import tqdm


class Iterator:

    def __init__(self, model: Model, dataset: DataSet):
        self.dataset = dataset
        self.model = model
        self.n = self.dataset.get_height()
        self.m = self.dataset.get_width()

    def get_model(self):
        return self.model

    def get_block(self, height, width):
        batchs = self.generate_batchs(height, self.n)
        for batch in tqdm(batchs):
            yield len(batch), self.get_block_activations(batch, width)

    def get_block_activations(self, batch_vertical, width):
        batchs = self.generate_batchs(width, self.m)
        for batch in batchs:
            x = self.dataset.get_matrix(batch_vertical, batch)
            x = self.model.predict(x)
            yield len(batch), x

    def generate_batchs(self, size_batch, n):
        aux = range(n)
        h = int(n/size_batch)
        batchs = [aux[j*size_batch:(j+1)*size_batch] for j in range(h)]
        if(n % size_batch > 0):
            batchs.append(aux[h*size_batch:n])
            h = h+1
        return batchs
