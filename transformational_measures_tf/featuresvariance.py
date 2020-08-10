import tensorflow as tf
from iterator import Iterator
from time import time
from variance import Variance
from metric import Metric


class FeaturesVariance(Metric):

    def __init__(self, iterator: Iterator = None):
        if iterator != None:
            self.set_iterator(iterator)

        self.result = None

    # overriding abstract method
    def set_iterator(self, iterator: Iterator):
        self.iterator = iterator
        self.variance = Variance(iterator)
        self.model_structure = self.iterator.get_model().model_structure

    # overriding abstract method
    def get_result(self):
        return self.result

    # overriding abstract method
    def compute(self, height, width):
        self.variance.compute(height, width)
        self.variance_layers_activations = self.variance.variance_layers_activations

        self.variance_layers_features = []
        aux = []
        for k in self.model_structure.map_layers_features:

            layer = self.variance_layers_activations[k]
            variance_features = []

            for f in range(self.model_structure.layers[k].number_features):

                feature = tf.unstack(layer, axis=tf.rank(layer)-1)[f]
                variance_features.append(tf.reduce_mean(feature))

            aux.append(tf.reduce_mean(tf.stack(variance_features)))

            self.variance_layers_features.append(variance_features)

        self.total_variance = tf.reduce_mean(tf.stack(aux))
        self.result = (self.total_variance, self.variance_layers_features,
                       self.variance_layers_activations)
        return self.result

    def get_feature(self, layer, feature):
        layer = self.variance_layers_activations[layer]
        feature = tf.unstack(layer, axis=tf.rank(layer)-1)[feature]
        return feature
