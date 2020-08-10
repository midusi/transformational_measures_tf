import tensorflow as tf
from iterator import Iterator
from time import time
from variance import Variance
from metric import Metric
from transformationsdataset import TransformationsDataSet
from variance import VarianceCalculations


class SameEquivariance:

    def __init__(self, iterator: Iterator = None, dataset: TransformationsDataSet = None):
        if iterator != None:
            self.set_iterator(iterator)
        if dataset != None:
            self.set_dataset(dataset)

        self.result = None

    # overriding abstract method
    def set_iterator(self, iterator: Iterator):
        self.iterator = iterator
        self.model_structure = self.iterator.get_model().model_structure

    def set_dataset(self, dataset: TransformationsDataSet):
        self.dataset = dataset

    # overriding abstract method
    def get_result(self):
        return self.result

    # overriding abstract method
    def compute(self, height, width):

        initial_time = time()
        calculations = VarianceCalculations(self.model_structure)

        for memory, blocks in self.iterator.get_block(height, width):
            calculations.renew(memory)
            for length_mov, block in blocks:

                if block.is_dataset_transpose() == False:

                    for k in self.model_structure.map_layers_features:
                        j = 0
                        for t in block.get_columns_dataset():
                            for i in range(block.height):
                                for f in range(self.model_structure.layers[k].number_features):
                                    aux = block[(i, j, k, f)]
                                    block[(i, j, k, f)] = self.dataset.apply_inverse_transformation(
                                        block[(i, j, k, f)], t)
                            j = j+1
                else:
                    for k in self.model_structure.map_layers_features:
                        j = 0
                        for t in block.get_rows_dataset():
                            for i in range(block.width):
                                for f in range(self.model_structure.layers[k].number_features):
                                    aux = block[(j, i, k, f)]
                                    block[(j, i, k, f)] = self.dataset.apply_inverse_transformation(
                                        block[(j, i, k, f)], t)
                            j = j+1

                calculations.add(block, length_mov)
            calculations.update()

        calculations.finish()

        self.variance_layers = calculations.variance_layers
        self.variance_layers_activations = calculations.variance_layers_activations

        self.variance_layers_features = []
        aux = []
        for k in self.model_structure.map_layers_features:

            layer = self.variance_layers_activations[k]
            variance_features = []

            for f in range(self.model_structure.layers[k].number_features):

                feature = tf.unstack(layer, axis=tf.rank(layer)-1)[f]
                variance_features.append(tf.reduce_mean(feature))

            self.variance_layers_features.append(variance_features)
            aux.append(tf.reduce_mean(tf.stack(variance_features)))

        self.time = time() - initial_time

        self.total_variance = tf.reduce_mean(tf.stack(aux))
        self.result = (self.total_variance, self.variance_layers_features,
                       self.variance_layers_activations)
        return self.result
