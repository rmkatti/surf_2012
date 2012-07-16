# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import HelmholtzMachine
from mnist import read_mnist
from util import prepare_mnist_images, shuffled_iter


def learn(digits = None, data_path = None, klass = None):
    digits = digits or range(10)
    klass = klass or HelmholtzMachine

    imgs, labels = read_mnist(path=data_path, training=True)
    idx = np.in1d(labels, digits)
    imgs = prepare_mnist_images(imgs[idx])
    labels = labels[idx]

    world = shuffled_iter(imgs)
    machine = klass(topology = (4, 128, 128, 28*28))
    machine.train(world.next, 
                  epsilon = 0.01, 
                  maxiter = 50000 * len(digits))
    return machine


if __name__ == '__main__':
    import argparse
    from neural.serialize import save

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs=1, metavar='OUTFILE')
    parser.add_argument('--laddered', action='store_true', default=False,
                        help='use the laddered Helmholtz machine')
    args = parser.parse_args()
    klass = None
    if args.laddered:
        from neural.helmholtz_laddered import LadderedHelmholtzMachine
        klass = LadderedHelmholtzMachine

    machine = learn(klass = klass)
    save(args.filenames[0], machine)
