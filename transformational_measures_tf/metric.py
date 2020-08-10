from abc import ABC, abstractmethod
from iterator import Iterator


class Metric(ABC):
    """
    An abstract class thats represents the main behavior of a Metric
    ...

    Attributes
    ----------

    Methods
    -------
    compute(height:int,width:int)
        computes the metric iterating over a block with such height and width
    set_iterator(iterator:Iterator)
        set the iterator in the metric
    get_result()
        returns the results computed by the metric

    """

    # abstract method
    def compute(self, height, width):
        pass

    # abstract method
    def set_iterator(self, iterator: Iterator):
        pass

    # abstract method
    def get_result(self):
        pass
