import time
import functools
import logging


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        out = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        logging.debug('function [{}] finished in {} ms.'.format(
            func.__name__, int(elapsedTime * 1000)))
        return out
    return newfunc