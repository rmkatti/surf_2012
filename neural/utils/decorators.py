# Standard library imports.
import functools


def memoize(obj):
    """ Memoize the results of a callable.

    Positional arguments must be hash-able and keyword arguments are not
    supported.
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args):
        if args not in cache:
            cache[args] = obj(*args)
        return cache[args]

    return memoizer
