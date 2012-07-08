# System library imports
import numpy as np
import matplotlib.pyplot as plt


def unit_vis(layers):
    """ Visualize the activity of the units in a neural network.

    Parameters:
    -----------
    layers : sequence of 2-dimensional array-like
        The layer activities of the neural net in top-to-bottom order.
    """
    plt.figure()
    for i, layer in enumerate(layers):
        layer = np.array(layer, copy=0)
        ax = plt.subplot(len(layers), 1, i+1)
        ax.matshow(layer, cmap=plt.cm.gray)
    plt.show()
