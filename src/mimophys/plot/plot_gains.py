from typing import Iterable, Optional, TypeVar

from matplotlib import pyplot as plt

AntennaArray = TypeVar("AntennaArray")


def plot_gains(
    arrays: Iterable[AntennaArray] | AntennaArray,
    weights: Optional[Iterable[float]] = [],
    **kwargs,
):
    """Plot the gain of multiple arrays.

    Parameters
    ----------
    *arrays : AntennaArray
        List of arrays to be plotted.
    weights : Iterable[float], optional
        Weights of the arrays. If None, all arrays will have the same weight.
    """

    if not isinstance(arrays, Iterable):
        arrays = [arrays]

    if len(weights) < len(arrays):
        weights += [None] * (len(arrays) - len(weights))

    figsize = kwargs.pop("figsize", (8, 8 * len(arrays)))
    fig, ax = plt.subplots(
        ncols=len(arrays),
        subplot_kw={"projection": "polar"},
        figsize=figsize,
    )

    if not isinstance(ax, Iterable):
        ax = [ax]

    for i, (array, weight) in enumerate(zip(arrays, weights)):
        array.plot_gain(weights=weight, ax=ax[i], **kwargs)

    plt.tight_layout()
    plt.show()


def plot_gains_3d(
    arrays: Iterable[AntennaArray] | AntennaArray,
    weights: Iterable[float] = [],
    **kwargs,
):
    """Plot the gain of multiple arrays.

    Parameters
    ----------
    *arrays : AntennaArray
        List of arrays to be plotted.
    weights : Iterable[float], optional
        Weights of the arrays. If None, all arrays will have the same weight.
    """

    if not isinstance(arrays, Iterable):
        arrays = [arrays]

    if len(weights) < len(arrays):
        weights += [None] * (len(arrays) - len(weights))

    figsize = kwargs.pop("figsize", (8 * len(arrays), 8))
    fig, ax = plt.subplots(
        ncols=len(arrays),
        subplot_kw={"projection": "3d"},
        figsize=figsize,
    )

    if not isinstance(ax, Iterable):
        ax = [ax]

    for i, (array, weight) in enumerate(zip(arrays, weights)):
        array.plot_gain_3d(weights=weight, ax=ax[i], **kwargs)
    plt.show()
