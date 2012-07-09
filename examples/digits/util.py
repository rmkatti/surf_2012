import numpy as np


def prepare_mnist_images(imgs):
    """ Prepare MNIST images for use in a binary neural network.
    """
    imgs = imgs.reshape((imgs.shape[0], 28*28)).astype(float)
    imgs /= 255.0
    return np.round(imgs, out=imgs)


def shuffled_iter(items):
    """ Iterate through items infinitely, shuffling them after every pass.
    """
    items = np.array(items)
    while True:
        np.random.shuffle(items)
        for item in items:
            yield item
