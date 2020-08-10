from model import Model
from transformationsdataset import TransformationsDataSet
from metric import Metric
from iterator import Iterator
import matplotlib.pyplot as plt


class Analyzer:

    """
    A class that analyzes the metric results vs the accuracy

    ...

    Attributes
    ----------

    Methods
    -------
    run(n:int)



    """

    def __init__(self, model: Model, dataset: TransformationsDataSet, metric: Metric):
        """
        Parameters
        ----------
        model: Model
        dataset : DataSet
        metric: Metric
        """

        self.model = model
        self.dataset = dataset
        self.metric = metric

    def run(self, n):
        """
        Parameters
        ----------
        n: int
            the amount of transformations in each group
        """

        self.x = []
        self.y = []

        for i in range(self.dataset.get_size_transformations_groups()):
            self.dataset.set_transformation_group(i, n)
            iterator = Iterator(self.model, self.dataset)
            self.metric.set_iterator(iterator)
            var = self.metric.compute(10, 5)[0]
            self.x.append(var)
            w = self.dataset.get_testing_data()
            x1 = w[0]
            y1 = w[1]

            acc = self.model.evaluate(x1, y1)[1]
            self.y.append(acc)

        plt.figure()
        plt.plot(self.x, self.y, 'ro')
