import json
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def make_weight_file(filebasename, weights, dir="./", other_data=None):
    """ Creates Weight File for

    Parameters
    ----------
    filebasename: str,
        output file name (no extension)
    weights: list of floats
        weight of each sequence in a separate fasta file, must be in same order as fasta file
    extension: str
        additional str specifier added to model names to indicate the model was trained with this
        corresponding weight file
    dir: str, optional, default="./"
        directory that weight file is saved to

    """
    with open(f'{dir}{filebasename}.json', 'w') as f:
        data = {"weights": weights}
        if other_data:
            data.update(other_data)
        json.dump(data, f)


def summary_np(nparray):
    """ Quick summary of 1D numpy array values

    Parameters
    ----------
    nparray: numpy array,
        1D numpy array of floats

    Returns
    -------
    summary: str,
        string with comma separated max of array, min or array, mean of array, and median of array
    """
    return f"{nparray.max()}, {nparray.min()}, {nparray.mean()}, {np.median(nparray)}"


def log_scale(listofnumbers, eps=1, base=math.e):
    """ Return log(x+base) for each number x in provided list

    Parameters
    ----------
    listofnumbers: list,
        list of float values
    eps: float, optional, default=1
        value added to each log operation to avoid 0 values ex. log(1) = 0
    base: float, optional, default=1
        value added to each log operation to avoid 0 values ex. log(1) = 0

    Returns
    -------
    data: np array
        log of value+base for value in parameter listofnumbers
    """
    return np.asarray([math.log(x + eps, base) for x in listofnumbers])


def quick_hist(x, outfile=None, yscale="log", bins=100):
    """ Make histogram of values, and save to file

    Parameters
    ----------
    x: list,
        values to make histogram with
    outfile: str,
        filename histogram is saved too, do not include file extension
    yscale: str, optional, default="log"
        set scale of yscale (matplotlib subplot scale options supported)
    bins: int, optional, default=100
        number of bins in histogram

    """
    fig, axs = plt.subplots(1, 1)
    axs.hist(x, bins=bins)
    axs.set_yscale(yscale)
    if outfile is not None:
        plt.savefig(outfile + ".png")
    else:
        plt.show()
    plt.close()


def scale_values_np(vals, min=0.05, max=0.95):
    """ Scale provided 1D np array values to b/t min and max

    Parameters
    ----------
    vals: np array
        1D np array of floats
    min: float, optional, default=0.05
        lowest value of vals will be scaled to this value
    max: flaot, optional, default=0.95
        largest value of vals will be scaled to this vales

    Returns
    -------
    data: np array
        np array of scaled values
    """
    nscaler = MinMaxScaler(feature_range=(min, max))
    return nscaler.fit_transform(vals.reshape(-1, 1))


def weight_transform(copy_nums, max_cutoff=None, exponent_base=math.e, exponent_min=-1, exponent_max=1):
    if max_cutoff is not None:
        return np.power(exponent_base, scale_values_np(log_scale([x if x < max_cutoff else max_cutoff for x in copy_nums], eps=1.0), min=exponent_min, max=exponent_max).squeeze(1))
    else:
        return np.power(exponent_base, scale_values_np(log_scale(copy_nums, eps=1.0), min=exponent_min, max=exponent_max).squeeze(1))
