import tensorflow as tf
from time import time
from transformational_measures_tf.iterator import Iterator
from transformational_measures_tf.metric import Metric
from transformational_measures_tf.sameequivariance import SameEquivariance
from transformational_measures_tf.transformationsdataset import TransformationsDataSet


class NormalizedSameEquivariance(Metric):

    def __init__(self, iterator: Iterator = None, dataset: TransformationsDataSet = None):
        self.iterator = iterator
        self.dataset = dataset
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

        sameequivariance = SameEquivariance(self.iterator, self.dataset)
        sameequivariance.compute(height, width)

        transformational_sameequivariance_layers_features = sameequivariance.variance_layers_features
        transformational_sameequivariance_layers_activations = sameequivariance.variance_layers_activations
        transformational_sameequivariance_total_variance = sameequivariance.total_variance

        self.iterator.transpose()

        sameequivariance.compute(height, width)

        sample_sameequivariance_layers_features = sameequivariance.variance_layers_features
        sample_sameequivariance_layers_activations = sameequivariance.variance_layers_activations
        sample_sameequivariance_total_variance = sameequivariance.total_variance

        n = len(transformational_sameequivariance_layers_features)

        self.variance_layers_features = []
        self.variance_layers = []
        for k in range(n):

            m = len(transformational_sameequivariance_layers_features[k])
            aux = []
            for f in range(m):
                aux.append(
                    transformational_sameequivariance_layers_features[k][f]/sample_sameequivariance_layers_features[k][f])

            self.variance_layers_features.append(aux)
            self.variance_layers.append(tf.math.reduce_mean(tf.stack(aux)))

        self.total_variance = transformational_sameequivariance_total_variance / \
            sample_sameequivariance_total_variance

        self.result = (self.total_variance, self.variance_layers,
                       self.variance_layers_features)

        self.time = time() - initial_time

        return self.result
