from matplotlib import pyplot as plt

from ..devices import AntennaArray


def plot_arrays(
    *arrays: AntennaArray, plane="xy", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple arrays in 2D projection.

    Args:
        *arrays (AntennaArray): List of arrays to be plotted.
        plane (str): Plane to plot in. Can be 'xy', 'yz' or 'xz'.
        **kwargs: Additional arguments to pass to the plotting function.

    Returns:
        tuple: Figure and axes objects.
    """
    fig, ax = plt.subplots(**kwargs)
    if plane == "xy":
        for array in arrays:
            ax.scatter(
                array.coordinates[:, 0],
                array.coordinates[:, 1],
                marker=array.marker,
                label=array.name,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif plane == "yz":
        for array in arrays:
            ax.scatter(
                array.coordinates[:, 1],
                array.coordinates[:, 2],
                marker=array.marker,
                label=array.name,
            )
        ax.set_xlabel("y")
        ax.set_ylabel("z")
    elif plane == "xz":
        for array in arrays:
            ax.scatter(
                array.coordinates[:, 0],
                array.coordinates[:, 2],
                marker=array.marker,
                label=array.name,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    else:
        raise ValueError("plane must be 'xy', 'yz' or 'xz'")

    ax.grid(True)
    ax.set_title(r"AntennaArray Projection in {}-plane".format(plane))
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_arrays_3d(*arrays, **kwargs):
    """Plot multiple arrays in 3D.

    Args:
        *arrays (AntennaArray): List of arrays to be plotted.
        **kwargs: Additional arguments to pass to the plotting function.

    Returns:
        tuple: Figure and axes objects.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for array in arrays:
        ax.scatter(
            array.coordinates[:, 0],
            array.coordinates[:, 1],
            array.coordinates[:, 2],
            marker=array.marker,
            label=array.name,
            **kwargs,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax
