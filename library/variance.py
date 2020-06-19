import tensorflow as tf
from iterator import Iterator
from time import time


class Variance:

    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.number_layers = self.iterator.get_model().size_layers()
        self.number_activations_layers = self.iterator.get_model().size_activations_layers()

    def compute(self, height, width):

        initial_time = time()
        calculations = Variance_Calculations(
            self.number_layers, self.number_activations_layers)

        for memory, blocks in self.iterator.get_block(height, width):
            calculations.renew(memory)
            for length_mov, block in blocks:
                calculations.add(block, length_mov)
            calculations.update()

        calculations.finish()


        self.variance_layers = calculations.variance_layers
        self.variance_layers_activations = calculations.variance_layers_activations

        self.time = time() - initial_time

        return (self.variance_layers,self.variance_layers_activations)


class Variance_Calculations:

    def __init__(self, number_layers, number_activations_layers):
        self.number_activations_layers = number_activations_layers
        self.number_layers = number_layers
        self.divisor_number = 0  
        self.variance_layers_activations = [tf.zeros(
            [self.number_activations_layers[r]], dtype=tf.dtypes.float32) for r in range(self.number_layers)]

    def renew(self, memory_length):
        self.count = 0
        self.memory_length = memory_length
        self.mean = [[tf.zeros([self.number_activations_layers[r]], dtype=tf.dtypes.float32)
                      for r in range(self.number_layers)] for q in range(memory_length)]
        self.moment = [[tf.zeros([self.number_activations_layers[r]], dtype=tf.dtypes.float32)
                        for r in range(self.number_layers)] for q in range(memory_length)]
        self.divisor_number += memory_length

    def add(self, block, mov_length):
        self.mov_length = mov_length
        for q in range(self.mov_length):
            self.count += 1
            for r in range(self.memory_length):

                for l in range(self.number_layers):

                    layer = block[(r, q, l)]

                    last_mean = self.mean[r][l]
                    self.mean[r][l] = tf.add(last_mean, tf.subtract(
                        layer, last_mean)/self.count)
                    self.moment[r][l] = tf.add(self.moment[r][l], tf.multiply(
                        tf.subtract(layer, last_mean), tf.subtract(layer, self.mean[r][l])))

    def update(self):
        for r in range(self.memory_length):
            self.variance_layers_activations = [tf.add(self.variance_layers_activations[l], (
                self.moment[r][l]/(self.count-1))) for l in range(self.number_layers)]

    def finish(self):
        self.variance_layers_activations = [
            self.variance_layers_activations[l]/self.divisor_number for l in range(self.number_layers)]
        self.variance_layers = [tf.reduce_mean(
            self.variance_layers_activations[l]) for l in range(self.number_layers)]
