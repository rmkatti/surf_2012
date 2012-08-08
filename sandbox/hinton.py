""" Draws Hinton diagrams using matplotlib.

Hinton diagrams visualize weight matrices, using color to denote sign and area
to denote magnitude.

By David Warde-Farley -- user AT cs dot toronto dot edu (user = dwf)
  with thanks to Geoffrey Hinton for providing the MATLAB code off of 
  which this is modeled.

Redistributable under the terms of the 3-clause BSD license 
(see http://www.opensource.org/licenses/bsd-license.php for details).
"""

import numpy as np
import matplotlib.pyplot as plt

def hinton(W, max_weight = None):
    """ Draws a Hinton diagram for visualizing a weight matrix. 
    """
    if max_weight is None:
        max_weight = 2 ** np.ceil(np.log(np.max(np.abs(W))) / np.log(2))

    # Temporarily disable matplotlib interactive mode if it is on, 
    # otherwise this takes forever.
    isinteractive = plt.isinteractive()
    if isinteractive:
        plt.ioff()

    height, width = W.shape        
    plt.clf()
    plt.fill(np.array([0, width, width, 0]), 
             np.array([0, 0, height, height]), 'gray')
    
    plt.axis('off')
    plt.axis('equal')
    for (y, x), w in np.ndenumerate(W):
        if w != 0:
            color = 'white' if w > 0 else 'black'
            area = min(1, np.abs(w) / max_weight)
            _blob(x + 0.5, height - y - 0.5, area, color)

    if isinteractive:
        plt.ion()

def _blob(x, y, area, color):
    """ Draws a square-shaped blob with the given area (< 1) at the given
    coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, color, edgecolor=color)

    
if __name__ == "__main__":
    hinton(np.random.randn(20, 20))
    plt.title('Example Hinton diagram - 20x20 random normal')
    plt.show()
