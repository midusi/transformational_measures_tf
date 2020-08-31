import tensorflow as tf
from time import time
from transformational_measures_tf.iterator import Iterator
from transformational_measures_tf.metric import Metric
from transformational_measures_tf.variance import Variance


class NormalizedVariance(Metric):

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

        variance = Variance(self.iterator)
        variance.compute(height, width)

        transformational_variance_layers = variance.variance_layers
        transformational_variance_layers_activations = variance.variance_layers_activations
        transformational_variance_total_variance = variance.total_variance

        self.iterator.transpose()

        variance.compute(height, width)

        sample_variance_layers = variance.variance_layers
        sample_variance_layers_activations = variance.variance_layers_activations
        sample_variance_total_variance = variance.total_variance

        n = len(sample_variance_layers)

        self.variance_layers = [
            transformational_variance_layers[i]/sample_variance_layers[i] for i in range(n)]
        self.variance_layers_activations = [tf.math.divide(
            transformational_variance_layers_activations[i], sample_variance_layers_activations[i]) for i in range(n)]
        self.total_variance = transformational_variance_total_variance / \
            sample_variance_total_variance

        self.result = (self.total_variance, self.variance_layers,
                       self.variance_layers_activations)

        self.time = time() - initial_time

        return self.result
