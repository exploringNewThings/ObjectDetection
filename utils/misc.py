import os
import logging
import pandas as pd
from .timing import timeit


@timeit
def writedataframe(df, savename=None):
    assert savename is not None, "No destination for saving the data frame " \
                                 "has been provided."

    logging.debug("Writing the DataFrame to {}".format(savename))
    df.to_csv(savename)
    return None