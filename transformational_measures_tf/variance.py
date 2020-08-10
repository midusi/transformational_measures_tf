import tensorflow as tf
from iterator import Iterator
from time import time
from metric import Metric


class Variance(Metric):

    def __init__(self, iterator: Iterator = None):
        self.iterator = iterator
        self.result = None

    # overriding abstract method
    def set_iterator(self, iterator: Iterator):
        self.iterator = iterator

    # overriding abstract method
    def get_result(self):
        return self.result

    # overriding abstract method
    def compute(self, height, width):

        initial_time = time()
        calculations = VarianceCalculations(
            self.iterator.get_model().model_structure)

        for memory, blocks in self.iterator.get_block(height, width):
            calculations.renew(memory)
            for length_mov, block in blocks:
                calculations.add(block, length_mov)
            calculations.update()

        calculations.finish()

        self.variance_layers = calculations.variance_layers
        self.variance_layers_activations = calculations.variance_layers_activations
        self.total_variance = tf.reduce_mean(tf.stack(self.variance_layers))

        self.result = (self.total_variance, self.variance_layers,
                       self.variance_layers_activations)

        self.time = time() - initial_time

        return self.result


class VarianceCalculations:

    def __init__(self, model_structure):
        self.model_structure = model_structure
        self.divisor_number = 0
        self.variance_layers_activations = [tf.zeros(
            shape=self.model_structure.layers[k].shape, dtype=tf.dtypes.float32) for k in range(self.model_structure.number_layers)]

    def renew(self, memory_length):
        self.count = 0
        self.memory_length = memory_length
        self.mean = [[tf.zeros(shape=self.model_structure.layers[k].shape, dtype=tf.dtypes.float32)
                      for k in range(self.model_structure.number_layers)] for q in range(memory_length)]
        self.moment = [[tf.zeros(shape=self.model_structure.layers[k].shape, dtype=tf.dtypes.float32)
                        for k in range(self.model_structure.number_layers)] for q in range(memory_length)]

        self.divisor_number += memory_length

    def add(self, block, mov_length):
        self.mov_length = mov_length
        for q in range(self.mov_length):
            self.count += 1
            for r in range(self.memory_length):

                for k in range(self.model_structure.number_layers):

                    layer = block[(r, q, k)]
                    last_mean = self.mean[r][k]
                    self.mean[r][k] = tf.add(
                        last_mean, tf.subtract(layer, last_mean)/self.count)
                    self.moment[r][k] = tf.add(self.moment[r][k], tf.multiply(
                        tf.subtract(layer, last_mean), tf.subtract(layer, self.mean[r][k])))

    def update(self):
        for r in range(self.memory_length):
            self.variance_layers_activations = [tf.add(self.variance_layers_activations[k], (
                self.moment[r][k]/(self.count-1))) for k in range(self.model_structure.number_layers)]

    def finish(self):
        self.variance_layers_activations = [self.variance_layers_activations[k] /
                                            self.divisor_number for k in range(self.model_structure.number_layers)]
        self.variance_layers = [tf.reduce_mean(
            self.variance_layers_activations[k]) for k in range(self.model_structure.number_layers)]
