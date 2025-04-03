from typing import Tuple

import numpy as np

from ..devices.antenna_array import AntennaArray

__all__ = ["relative_position"]


def relative_position(tx, rx) -> Tuple[float, float, float]:
    """Returns the relative position (range, azimuth and elevation) from loc1 to loc2.

    Parameters
    ----------
    loc1, loc2: array_like, shape (3,)
        Location of the 2 points.

    Returns
    -------
    range: float
        Distance between the 2 points.
    az: float
        Azimuth angle.
    el: float
        Elevation angle.
    """
    if isinstance(tx, AntennaArray):
        tx = tx.array_center
    if isinstance(rx, AntennaArray):
        rx = rx.array_center

    tx = np.asarray(tx).reshape(3)
    rx = np.asarray(rx).reshape(3)
    dxyz = dx, dy, dz = rx - tx
    r = np.linalg.norm(dxyz)
    az = np.arctan2(dx, dy)
    el = np.arcsin(dz / r)
    return r, az, el


def sph2cart(r, az, el):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r: float
        Radial distance.
    az: float
        Azimuthal angle.
    el: float
        Elevation angle.

    Returns
    -------
    x, y, z: float
        Cartesian coordinates.
    """
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return x, y, z