import numpy as np
from typing import Tuple

__all__ = ["relative_position"]


def relative_position(loc1, loc2) -> Tuple[float, float, float]:
    """Returns the relative position (range, azimuth and elevation) between 2 locations.

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
    loc1 = np.asarray(loc1).reshape(3)
    loc2 = np.asarray(loc2).reshape(3)
    dxyz = dx, dy, dz = loc2 - loc1
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