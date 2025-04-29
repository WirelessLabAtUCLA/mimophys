from typing import Tuple

import numpy as np

__all__ = ["relative_position", "sph2cart", "rotation_matrix"]


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
    tx = np.asarray(tx).reshape(3)
    rx = np.asarray(rx).reshape(3)
    dx, dy, dz = tx - rx
    r = np.sqrt(dx**2 + dy**2 + dz**2)
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


def rotation_matrix(axis, theta) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    

    Args:
        axis (array-like): The axis of rotation. Should be a 3-element array-like object.
        theta (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotation matrix as a 3x3 numpy array.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
