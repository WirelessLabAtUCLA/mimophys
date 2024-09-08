from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from matplotlib import cm
from numpy import log10


class AntennaArray:
    """Base class for array objects.

    Parameters
    ----------
    num_antennas : int
        Number of antennas in the array.
    coordinates : array_like
        Coordinates of the antennas. The shape of the array must be (num_antennas, 3).
    weights : array_like, optional
        Weights of the antennas. If not given, all antennas are assumed to have
        unit weight.
    """

    def __init__(self, N: int, coordinates: Iterable = [0, 0, 0], **kwargs):
        self.num_antennas = N
        self.coordinates = np.array(coordinates)
        self.weights = np.ones(N)
        self.marker = "o"
        self.power = 1
        self.noise_db = 0
        self.name = "AntennaArray"
        self.spacing = 0.5
        self.frequency = 1e9
        self._config = f"({N} elm)"
        for kwarg in kwargs:
            self.__setattr__(kwarg, kwargs[kwarg])

    N = Nr = Nt = property(lambda self: self.num_antennas)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name + " " + self._config

    def __len__(self):
        return self.num_antennas

    @property
    def amp(self):
        return np.abs(self.weights)

    @property
    def phase(self):
        return np.angle(self.weights)

    @property
    def array_center(self):
        """Returns the center of the array."""
        return np.mean(self.coordinates, axis=0)

    @array_center.setter
    def array_center(self, center):
        """Set the center of the array."""
        delta_center = center - self.array_center
        self.coordinates += delta_center

    @property
    def location(self):
        """Returns the location of the array. default is the center of the array."""
        return np.mean(self.coordinates, axis=0)

    @property
    def diameter(self):
        """Returns the diameter of the array."""
        Dx = np.max(self.coordinates[:, 0]) - np.min(self.coordinates[:, 0])
        Dy = np.max(self.coordinates[:, 1]) - np.min(self.coordinates[:, 1])
        Dz = np.max(self.coordinates[:, 2]) - np.min(self.coordinates[:, 2])
        return np.sqrt(Dx**2 + Dy**2 + Dz**2)

    @location.setter
    def location(self, location):
        """Set the location of the array."""
        delta_location = location - self.location
        self.coordinates += delta_location

    @property
    def noise(self):
        return 10 ** (self.noise_db / 10)

    @noise.setter
    def noise(self, noise):
        self.noise_db = 10 * log10(noise + np.finfo(float).tiny)

    @classmethod
    def ula(
        cls,
        N,
        array_center=[0, 0, 0],
        ax="x",
        spacing=0.5,
        **kwargs,
    ):
        """Empties the array and creates a half-wavelength spaced,
        uniforom linear array along the desired axis centered at the origin.

        Parameters
        ----------
        N : int
            Number of antennas in the array.
        ax : char, optional
            Axis along which the array is to be created.
            Takes value 'x', 'y' or 'z'. Default is 'x'.
        array_center : array_like, optional
            Coordinates of the center of the array. Default is [0, 0, 0].
        spacing : float, optional
            Spacing between the antennas. Default is 0.5.
        normalize : bool, optional
            If True, the weights are normalized to have unit norm. Default is True.
        """
        if ax == "x":
            coordinates = np.array([np.arange(N), np.zeros(N), np.zeros(N)]).T
        elif ax == "y":
            coordinates = np.array(
                [
                    np.zeros(N),
                    np.arange(N),
                    np.zeros(N),
                ]
            ).T
        elif ax == "z":
            coordinates = np.array(
                [
                    np.zeros(N),
                    np.zeros(N),
                    np.arange(N),
                ]
            ).T
        else:
            raise ValueError("ax must be 'x', 'y' or 'z'")
        ula = cls(N, coordinates * spacing)
        ula.array_center = array_center
        config_map = {"x": f"({N}11)", "y": f"(1{N}1)", "z": f"(11{N})"}
        ula._config = config_map[ax]
        for kwarg in kwargs:
            ula.__setattr__(kwarg, kwargs[kwarg])
        return ula

    initialize_ula = ula

    @classmethod
    def upa(
        cls,
        N: Iterable,
        array_center=(0, 0, 0),
        plane="xy",
        spacing=0.5,
        **kwargs,
    ):
        """Empties the array and creates a half-wavelength spaced,
        uniform plannar array in the desired plane.

        Parameters
        ----------
        N : Iterable
            Number of rows and columns in the array.
        plane : str, optional
            Plane in which the array is to be created or the axis orthogonal to the plane.
            Takes value 'xy', 'yz' or 'xz'. Default is 'xy'.
        array_center : array_like, optional
            Coordinates of the center of the array. Default is [0, 0, 0].
        normalize : bool, optional
            If True, the weights are normalized to have unit norm. Default is True.
        """
        num_rows = N[0]
        num_cols = N[1]
        if plane == "xy":
            coordinates = np.array(
                [
                    np.tile(np.arange(num_cols), num_rows),
                    np.repeat(np.arange(num_rows), num_cols),
                    np.zeros(num_rows * num_cols),
                ]
            ).T
        elif plane == "yz":
            coordinates = np.array(
                [
                    np.zeros(num_rows * num_cols),
                    np.tile(np.arange(num_cols), num_rows),
                    np.repeat(np.arange(num_rows), num_cols),
                ]
            ).T
        elif plane == "xz":
            coordinates = np.array(
                [
                    np.tile(np.arange(num_cols), num_rows),
                    np.zeros(num_rows * num_cols),
                    np.repeat(np.arange(num_rows), num_cols),
                ]
            ).T
        else:
            raise ValueError("plane must be 'xy', 'yz' or 'xz'")
        upa = cls(num_rows * num_cols, coordinates * spacing)
        upa.array_center = array_center
        for kwarg in kwargs:
            upa.__setattr__(kwarg, kwargs[kwarg])
        config_map = {
            "xy": f"({num_rows}{num_cols}1)",
            "yz": f"(1{num_rows}{num_cols})",
            "xz": f"({num_rows}1{num_cols})",
        }
        upa._config = config_map[plane]
        return upa

    initialize_upa = upa

    @classmethod
    def from_file(cls, filename):
        """Load an array from a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the array from.
        """
        raise NotImplementedError

    def to_file(self, filename):
        """Save the array to a file.

        Parameters
        ----------
        filename : str
            Name of the file to save the array to.
        """
        np.save(filename, [self.coordinates, self.weights, self.marker])

    def reset(self):
        """reset weights to 1"""
        self.weights = np.ones(self.num_antennas)

    def normalize_weights(self, norm=1):
        """Normalize the weights of the antennas to have unit norm."""
        if LA.norm(self.weights) != 0:
            self.weights = self.weights * norm / LA.norm(self.weights)

    def set_weights(self, weights: Iterable | complex):
        """Set the weights of the antennas.

        Parameters
        ----------
        weights : array_like or float
            Weights of the antennas.
            If an array is given, the shape of the array must match the length of coordinates given.
            If a float is given, all antennas are changed to the same weight. and coordinates are ignored.
        index : array_like, optional
            Indices of the antennas whose weight is to be changed. If not given, the
            weights of all antennas are passed.
        normalize : bool, optional
            If True, the weights are normalized to have unit norm. Default is True.
        """

        if np.isscalar(weights):
            self.weights = weights * np.ones(self.num_antennas)
        else:
            weights = np.asarray(weights).reshape(-1)
            if len(weights) != self.num_antennas:
                raise ValueError(
                    "The length of weights must match the number of antennas"
                )
            self.weights = np.asarray(weights).reshape(-1)

    def get_weights(self, coordinates=None):
        """Get the weights of the antennas.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas whose weight is to be changed. If not
            given, the coordinates of all antennas are passed.
        """
        if coordinates is None:
            return self.weights
        else:
            indices = self._match_coordinates(coordinates)
            print(indices)
            if len(indices) == 0:
                raise ValueError("No matching coordinates found")
            return self.weights[indices]

    def _match_coordinates(self, coordinates):
        """Match the given coordinates to the coordinates of the array.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas to be matched. The shape of the array must be (num_antennas, 3).
        """

        # match each coordinate to with the coordinate in the array and return the indices
        indices = []
        coordinates = np.reshape(coordinates, (-1, 3))
        indices = np.where((coordinates[:, None] == self.coordinates).all(axis=2))[1]
        return indices

    ############################
    #  Antenna Manipulation
    ############################

    def add_elements(self, coordinates):
        """Add antennas to the array.

        Parameters
        ----------
        coordinates : array_like
            Coordinates of the antennas to be added. The shape of the array must be (num_antennas, 3).
        """
        self.coordinates = np.concatenate((self.coordinates, coordinates))
        self.num_antennas += coordinates.shape[0]
        self.weights = np.concatenate((self.weights, np.ones(coordinates.shape[0])))

    def remove_elements(self, coordinates=None, indices=None):
        """Remove antennas from the array by coordinates or indices.

        Parameters
        ----------
        coordinates : array_like, optional
            Coordinates of the antennas to be removed. The shape of the array must be (num_antennas, 3).
        indices : array_like, optional
            Indices of the antennas to be removed."""

        def _remove_elements_by_coord(self, coordinates):
            indices = self._match_coordinates(coordinates)
            self.coordinates = np.delete(self.coordinates, indices, axis=0)
            self.weights = np.delete(self.weights, indices, axis=0)
            self.num_antennas -= len(indices)

        def _remove_elements_by_idx(self, indices):
            self.coordinates = np.delete(self.coordinates, indices, axis=0)
            self.weights = np.delete(self.weights, indices, axis=0)
            self.num_antennas -= len(indices)

        if coordinates is not None:
            _remove_elements_by_coord(coordinates)
        elif indices is not None:
            _remove_elements_by_idx(indices)
        else:
            raise ValueError("Either coordinates or indices must be given")

    # @staticmethod
    # def _translate_coordinates(coordinates, shift=None):
    #     """Shift all elements of the array by the given coordinates.

    #     Parameters
    #     ----------
    #     coordinates: array_like
    #         Coordinates of the array to be shifted.
    #     shift: array_like, optional
    #         Coordinates by which the array is to be shifted. If not given, the
    #         array is centered at the origin.
    #     """
    #     if shift is None:
    #         shift = -np.mean(coordinates, axis=0)
    #     shift = np.asarray(shift).reshape(1, -1)
    #     coordinates += shift
    #     return coordinates

    # def translate(self, coordinates=None):
    #     """Shift all elements of the array by the given coordinates.
    #     Parameters
    #     ----------
    #     coordinates: array_like, optional
    #         Coordinates by which the array is to be shifted. If not given, the
    #         array is centered at the origin.
    #     """
    #     if coordinates is None:
    #         coordinates = -np.mean(self.coordinates, axis=0)
    #     self.coordinates += coordinates
    #     return coordinates

    def _rotate(self, coordinates, x_angle, y_angle, z_angle):
        """Rotate the array by the given angles.
        Parameters
        ----------
        x_angle : float
            Angle of rotation about the x-axis in radians.
        y_angle : float
            Angle of rotation about the y-axis in radians.
        z_angle : float
            Angle of rotation about the z-axis in radians.
        """
        rotation_matrix = np.array(
            [
                [
                    np.cos(y_angle) * np.cos(z_angle),
                    np.cos(z_angle) * np.sin(x_angle) * np.sin(y_angle)
                    - np.cos(x_angle) * np.sin(z_angle),
                    np.cos(x_angle) * np.cos(z_angle) * np.sin(y_angle)
                    + np.sin(x_angle) * np.sin(z_angle),
                ],
                [
                    np.cos(y_angle) * np.sin(z_angle),
                    np.cos(x_angle) * np.cos(z_angle)
                    + np.sin(x_angle) * np.sin(y_angle) * np.sin(z_angle),
                    -np.cos(z_angle) * np.sin(x_angle)
                    + np.cos(x_angle) * np.sin(y_angle) * np.sin(z_angle),
                ],
                [
                    -np.sin(y_angle),
                    np.cos(y_angle) * np.sin(x_angle),
                    np.cos(x_angle) * np.cos(y_angle),
                ],
            ]
        )

        translate_coordinates = self.translate()  # center the array at the origin
        self.coordinates = np.dot(coordinates, rotation_matrix)  # rotate the array
        self.translate(
            -translate_coordinates
        )  # translate the array back to its original position

        return np.dot(coordinates, rotation_matrix)

    def rotate(self, x_angle=0.0, y_angle=0.0, z_angle=0.0, inplace=True):
        """Rotate the array by the given angles.

        Parameters
        ----------
        x_angle : float
            Angle of rotation about the x-axis in radians.
        y_angle : float
            Angle of rotation about the y-axis in radians.
        z_angle : float
            Angle of rotation about the z-axis in radians.
        inplace : bool, optional
            If True, the array is rotated in-place. If False, a new array is
            returned. Default is True.
        """

        if inplace:
            self._rotate(self.coordinates, x_angle, y_angle, z_angle)
            return self
        else:
            new_array = self.copy()
            new_array._rotate(new_array.coordinates, x_angle, y_angle, z_angle)
            return new_array

    ############################
    # Get AntennaArray Properties
    ############################

    def get_array_response(self, az=0, el=0, torch_device=None, return_tensor=False):
        """Returns the array response vector at a given azimuth and elevation.

        This response is simply the phase shifts experienced by the elements
        on an incoming wavefront from the given direction, normalied to the first
        element in the array

        Parameters
        ----------
        az : float, array_like
            Azimuth angle in radians.
        el : float, array_like
            Elevation angle in radians.
        torch_device : str, optional
            If given, PyTorch is used to calculate the array response. Default is None.
        return_tensor : bool, optional
            If True, the array response is returned as a PyTorch tensor.
            Only valid if torch_device is given.
            Default is False.

        Returns
        -------
        array_response: The array response vector up to 3 dimensions. The shape of the array is
        (len(az), len(el), len(coordinates)) and is squeezed if az and/or el are scalars.
        """
        if torch_device is not None:
            return self._get_array_response_torch(az, el, torch_device, return_tensor)

        # calculate the distance of each element from the first element
        dx = self.coordinates[:, 0] - self.coordinates[0, 0]
        dy = self.coordinates[:, 1] - self.coordinates[0, 1]
        dz = self.coordinates[:, 2] - self.coordinates[0, 2]

        dx = np.expand_dims(dx, (0, 1))
        dy = np.expand_dims(dy, (0, 1))
        dz = np.expand_dims(dz, (0, 1))
        az = np.expand_dims(np.asarray(az).flatten(), (1, 2))
        el = np.expand_dims(np.asarray(el).flatten(), (0, 2))

        array_response = np.exp(
            1j
            * 2
            * np.pi
            * (
                dx * np.sin(az) * np.cos(el)
                + dy * np.cos(az) * np.cos(el)
                + dz * np.sin(el)
            )
        )
        array_response = np.squeeze(array_response)
        if self.num_antennas == 1:
            array_response = array_response.reshape(-1, 1)
        return array_response

    def _get_array_response_torch(self, az, el, device, return_tensor=False):
        """Use PyTorch to calculate number of responses in parallel."""
        from torch import as_tensor, cos, exp, sin
        from torch.cuda import empty_cache

        nc = self.coordinates
        dx = as_tensor(nc[:, 0] - nc[0, 0], device=device).reshape(1, 1, -1)
        dy = as_tensor(nc[:, 1] - nc[0, 1], device=device).reshape(1, 1, -1)
        dz = as_tensor(nc[:, 2] - nc[0, 2], device=device).reshape(1, 1, -1)

        az = as_tensor(az, device=device).reshape(-1, 1, 1)
        el = as_tensor(el, device=device).reshape(1, -1, 1)

        array_response = exp(
            (1j * 2 * np.pi)
            * (dx * sin(az) * cos(el) + dy * cos(az) * cos(el) + dz * sin(el))
        ).squeeze()
        if self.num_antennas == 1:
            array_response = array_response.reshape(-1, 1)

        if return_tensor:
            return array_response

        np_array_response = array_response.cpu().numpy()
        del array_response
        empty_cache()
        return np_array_response

    def get_array_gain(self, az, el, db=True, use_deg=True):
        """Returns the array gain at a given azimuth and elevation in dB.

        Parameters
        ----------
        az : float
            Azimuth angle in radians.
        el : float
            Elevation angle in radians.
        db : bool, optional
            If True, the gain is returned in dB. Default is True.

        Returns
        -------
        array_gain: The array gain at the given azimuth and elevation
            with shape (len(az), len(el))
        """

        if use_deg:
            az = az * np.pi / 180
            el = el * np.pi / 180

        array_response = self.get_array_response(az, el)
        # multiply gain by the weights at the last dimension
        gain = (array_response @ self.weights.conj().reshape(-1, 1)) ** 2
        gain = np.asarray(np.squeeze(gain))
        mag = np.abs(gain)
        # phase = np.angle(gain)
        # print(gain)
        if db:
            return 10 * log10(mag + np.finfo(float).tiny)
        return mag

    get_gain = get_array_gain

    def conjugate_beamformer(self, az=0, el=0):
        """Returns the conjugate beamformer at a given azimuth and elevation.

        Parameters
        ----------
        az : float
            Azimuth angle in degrees.
        el : float
            Elevation angle in degrees.
        """
        array_response_vector = self.get_array_response(
            az * np.pi / 180, el * np.pi / 180
        )
        return array_response_vector

    def get_array_pattern_azimuth(self, el, num_points=360, range=360):
        """Returns the array pattern at a given elevation.

        Parameters
        ----------
        el : float
            Elevation angle in radians.
        num_points : int, optional
            Number of points at which the pattern is to be calculated.
            Default is 360.
        range : float, optional
            Range of azimuth angles in degrees. Default is 360.
        """
        az = np.linspace(-range / 2, range / 2, num_points) * np.pi / 180
        return self.get_array_response(az, el)

    array_pattern_azimuth = get_array_pattern_azimuth

    ############################
    # Plotting
    ############################

    def plot_gain_el(self, cut=0, angles=np.linspace(-89, 89, 178), **kwargs):
        """Plot the array pattern at a given elevation."""
        return self.plot_gain(cut, angles, "el", **kwargs)

    def plot_gain_az(self, cut=0, angles=np.linspace(-89, 89, 178), **kwargs):
        """Plot the array pattern at a given azimuth."""
        return self.plot_gain(cut, angles, "az", **kwargs)

    def plot_gain(
        self,
        polar=True,
        cut=0,
        angles=np.linspace(-89, 89, 356),
        cut_along="el",
        weights=None,
        db=True,
        ax=None,
        ylim=-20,
        **kwargs,
    ):
        """Plot the array pattern at a given elevation or azimuth.

        Parameters
        ----------
        cut : float
            Elevation or azimuth angle in degrees. Angle at which the pattern is to be plotted.
        angles : array_like
            Azimuth or elevation angles in degrees.
        cut_along : str, optional
            Axis along which the cut is to be made. Takes value 'el' or 'az'. Default is 'el'.
        weights : array_like, optional
            Weights of the antennas. If not given, the weights are not changed.
        polar : bool, optional
            If True, the pattern is plotted in polar coordinates. Default is False.
        db : bool, optional
            If True, the gain is plotted in dB. Default is True.
        ax : matplotlib.axes.Axes, optional
            The matplotlib axes object. If not given, a new figure is created.
        **kwargs : optional
            matplotlib.pyplot.plot arguments.
        """
        if weights is not None:
            orig_weights = self.get_weights()
            self.set_weights(weights)
        if cut_along == "el":
            el = np.asarray(cut) * np.pi / 180
            az = np.asarray(angles) * np.pi / 180
        elif cut_along == "az":
            az = np.asarray(cut) * np.pi / 180
            el = np.asarray(angles) * np.pi / 180
        else:
            raise ValueError("cut_along must be 'el' or 'az'")

        # vectorized version
        gain = self.get_array_gain(az, el, db=db, use_deg=False)

        if ax is None:
            if polar:
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            else:
                fig, ax = plt.subplots()
        if polar:
            ax.plot(angles * np.pi / 180, gain, **kwargs)
            ax.set_theta_zero_location("E")
            # ax.set_theta_direction(-1)
            # ax.set_rlabel_position(-90)
            # ax.set_rticks([-20, -10, 0])
            # ax.set_rlim(-20, 0)
            # limit theta range to 180
            ax.set_thetamin(min(angles))
            ax.set_thetamax(max(angles))
            ax.set_ylabel("Gain (dB)")
            ax.set_xlabel("Azimuth (deg)")
            ax.set_theta_direction(-1)
        else:
            ax.plot(angles, gain, **kwargs)
            # ax.set_ylim(-(max(array_response)), max(array_response) + 10)
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Gain (dB)")
        cut_name = "el" if cut_along == "el" else "az"
        title = f"{cut_name} = {cut} deg, max gain = {np.max(np.real(gain)):.2f} dB"
        ax.set_title(title)
        ax.grid(True)
        if weights is not None:
            self.set_weights(orig_weights)
        if ax is None:
            plt.tight_layout()
            plt.show()
        return ax

    @staticmethod
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    def plot_gain_3d(
        self,
        az=np.linspace(-180, 180, 360),
        el=np.linspace(-90, 90, 180),
        ax=None,
        max_gain=None,
        min_gain=None,
        polar=False,
        **kwargs,
    ):
        gain = self.get_array_gain(az, el, db=True, use_deg=True)
        az_grid, el_grid = np.meshgrid(az, el)

        if max_gain is None:
            max_gain = np.max(gain)
        if min_gain is None:
            min_gain = np.min(gain)
        gain = np.clip(gain, min_gain, max_gain).T

        if polar:
            az_grid, el_grid, gain = self.cart2sph(az_grid, el_grid, gain)

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)
        colors = cm.YlGnBu_r(gain)
        ax.plot_surface(
            az_grid,
            el_grid,
            gain,
            cmap="magma",
            facecolors=colors,
            # linewidth=1,
        )
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_zlabel("Gain (dB)")
        if ax is None:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_array_3d(self, **kwargs):
        """Plot the array."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            self.coordinates[:, 2],
            marker=self.marker,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def plot_array(self, plane="xy", ax=None):
        """Plot the array in 2D projection

        Parameters
        ----------
        plane : str, optional
            Plane in which the array is to be projected.
            Takes value 'xy', 'yz' or 'xz'. Default is 'xy'.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if plane == "xy":
            ax.scatter(
                self.coordinates[:, 0], self.coordinates[:, 1], marker=self.marker
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif plane == "yz":
            ax.scatter(
                self.coordinates[:, 1], self.coordinates[:, 2], marker=self.marker
            )
            ax.set_xlabel("y")
            ax.set_ylabel("z")
        elif plane == "xz":
            ax.scatter(
                self.coordinates[:, 0], self.coordinates[:, 2], marker=self.marker
            )
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        else:
            raise ValueError("plane must be 'xy', 'yz' or 'xz'")
        ax.grid(True)
        ax.set_title(r"AntennaArray Projection in {}-plane".format(plane))

        if ax is None:
            plt.show()
        return ax

    def plot(self, **kwargs):
        """Plot the array."""
        return self.plot_array(**kwargs)

    def plot_3d(self, **kwargs):
        """Plot the array in 3D."""
        return self.plot_array_3d(**kwargs)
