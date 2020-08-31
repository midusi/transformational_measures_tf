import tensorflow as tf
from time import time
from transformational_measures_tf.iterator import Iterator
from transformational_measures_tf.metric import Metric
from transformational_measures_tf.featuresvariance import FeaturesVariance


class NormalizedFeaturesVariance(Metric):

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

        featuresvariance = FeaturesVariance(self.iterator)
        featuresvariance.compute(height, width)

        transformational_featuresvariance_layers_features = featuresvariance.variance_layers_features
        transformational_featuresvariance_layers_activations = featuresvariance.variance_layers_activations
        transformational_featuresvariance_total_variance = featuresvariance.total_variance

        self.iterator.transpose()

        featuresvariance.compute(height, width)

        sample_featuresvariance_layers_features = featuresvariance.variance_layers_features
        sample_featuresvariance_layers_activations = featuresvariance.variance_layers_activations
        sample_featuresvariance_total_variance = featuresvariance.total_variance

        n = len(transformational_featuresvariance_layers_features)

        self.variance_layers_features = []
        self.variance_layers = []
        for k in range(n):

            m = len(transformational_featuresvariance_layers_features[k])
            aux = []
            for f in range(m):
                aux.append(
                    transformational_featuresvariance_layers_features[k][f]/sample_featuresvariance_layers_features[k][f])

            self.variance_layers_features.append(aux)
            self.variance_layers.append(tf.math.reduce_mean(tf.stack(aux)))

        self.total_variance = transformational_featuresvariance_total_variance / \
            sample_featuresvariance_total_variance

        self.result = (self.total_variance, self.variance_layers,
                       self.variance_layers_features)

        self.time = time() - initial_time

        return self.result
