import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Graphs:
    """
    A class used to graph heatmaps
    ...

    Attributes
    ----------

    Methods
    -------
    hetmap_list2D(list2D:list,columns_name:list,title:str)

    heatmap_matrix(matrix,columns_name:list,title:str)

    """

    def __init__(self):
        pass

    def plot(self, x, y, title=None):
        plt.figure()
        plt.title(title)
        plt.plot(x, y)

    def heatmap_list2D(self, list2D: list, columns_names: list = None, title: str = None):
        """It Graphs a heatmap  

        Parameters
        ----------
        list2D:list
            a 2D list with the heatmap values
        columns_name:list
            it contains the names for each column in the heat map
        title:str
            the heatmap title

        """

        m = len(list2D)
        y = [tf.reduce_max(list2D[i]) for i in range(m)]
        vmax = tf.reduce_max(tf.stack(y))
        y = [tf.reduce_min(list2D[i]) for i in range(m)]
        vmin = tf.reduce_min(tf.stack(y))
        if(vmin < 0):
            vmin = 0

        f, axes = plt.subplots(1, m)
        for i in range(m):
            ax = axes[i]
            ax.axis("off")
            column = np.array(list2D[:][i])
            column = column[:, np.newaxis]

            mappable = ax.imshow(column, vmax=vmax, vmin=vmin,
                                 cmap='inferno', aspect="auto")

            if columns_names != None:
                name = columns_names[i]
            else:
                name = ""

            if len(name) > 7:
                name = name[:10]+"."

            ax.set_title(name, fontsize=8, rotation=45)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = f.colorbar(mappable, cax=cbar_ax, extend='max')
        cbar.cmap.set_over('green')
        cbar.cmap.set_bad(color='blue')
        if title != None:
            f.suptitle(title)

    def heatmap_matrix(self, matrix, columns_names: list = None, title: str = None):
        """It Graphs a heatmap  
        Parameters
        ----------
        matrix
            a 2D array representing a matrix with the heatmap values
        columns_name:list
            it contains the names for each column in the heat map
        title:str
            the heatmap title
            """

        matrix = tf.convert_to_tensor(matrix)
        shape = matrix.shape
        n = shape[0]
        m = shape[1]

        vmax = tf.reduce_max(matrix)
        vmin = tf.reduce_min(matrix)
        if(vmin < 0):
            vmin = 0

        f, axes = plt.subplots(1, m)
        for i in range(m):
            ax = axes[i]
            ax.axis("off")
            column = np.array(matrix[:, i])
            column = column[:, np.newaxis]

            mappable = ax.imshow(column, vmax=vmax, vmin=vmin,
                                 cmap='inferno', aspect="auto")

            if columns_names != None:
                name = columns_names[i]
            else:
                name = ""

            if len(name) > 7:
                name = name[:10]+"."

            ax.set_title(name, fontsize=8, rotation=45)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = f.colorbar(mappable, cax=cbar_ax, extend='max')
        cbar.cmap.set_over('green')
        cbar.cmap.set_bad(color='blue')

        if title != None:
            f.suptitle(title)
