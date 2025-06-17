from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from matplotlib import cm
from matplotlib.axes import Axes
from numpy import log10
from numpy.typing import ArrayLike

from ..plot import plot_arrays, plot_arrays_3d
from ..utils.geometry import rotation_matrix

"""Azimuth 0 is along the y-axis, elevation 0 is in the x-y plane. (update 2024.10.30)
    x = r * sin(theta) * cos(phi)
    y = r * cos(theta) * cos(phi)
    z = r * sin(phi)
    
    See https://ianproberts.com/notes/arrays.html
"""


class AntennaArray:
    """Base class for array objects.

    Args:
        N: Number of antennas in the array.
        coordinates: Coordinates of the antennas. The shape of the array must be (num_antennas, 3).
        power: Power level of the antenna array.
        noise_power: Noise power level.
        power_dbm: Power level in dBm (if specified, overrides power).
        noise_power_dbm: Noise power level in dBm (if specified, overrides noise_power).
        name: Name identifier for the antenna array.
        weights: Weights of the antennas. If not given, all antennas are assumed to have unit weight.
        frequency: Operating frequency in Hz.
        marker: Marker style for plotting.
    """

    def __init__(
        self,
        N: Optional[int | ArrayLike] = None,
        spacing: float = 0.5,
        array_center: ArrayLike = (0, 0, 0),
        coordinates: Optional[ArrayLike] = None,
        power: float = 1,
        noise_power: float = 0,
        dbm: bool = False,
        name: str = "AntennaArray",
        weights: ArrayLike | None = None,
        frequency: float = 1e9,
        marker: str = "o",
    ):
        """Initialize the antenna array.

        Args:
            N (int | Iterable): Number of antennas in the array.
                Each element corresponds to the number of antennas in [x, z, y] respectively.
                Empty elements are padded with ones.
                If `coordinates` is given, this argument is ignored.
            spacing (float): Spacing between the antennas. Default is 0.5.
                If `coordinates` is given, this argument is ignored.
            coordinates (Optional[ArrayLike]): Coordinates of the antennas to manually set each antenna.
                The shape of the array must be (num_antennas, 3).
            power (float): Power level of the antenna array. Default is 1.
            noise_power (float): Noise power level. Default is 0.
            dbm (bool): If True, power and noise_power are in dBm. Default is False.
            weights (ArrayLike | None): Weights of the antennas. If not given, all antennas are assumed to have unit weight.
            name (str): Name identifier for the antenna array. Default is "AntennaArray".
            frequency (float): Operating frequency in Hz. Default is 1e9.
            marker (str): Marker style for plotting. Default is "o".

        Raises:
            ValueError: If coordinates are not of shape (num_antennas, 3).
            ValueError: If neither coordinates nor N are given.
        """
        # Need to initialize coordinates first
        if coordinates is None:
            if N is None:
                raise ValueError("Either coordinates or N must be given")
            self.coordinates = self.grid_coord(N, spacing=spacing)
            self.array_center = array_center

        else:
            if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
                raise ValueError("Coordinates must be of shape (num_antennas, 3)")
            self.coordinates = np.array(coordinates)

        self.weights = np.ones(N) if weights is None else np.array(weights)
        self.name = name
        self.frequency = frequency
        self._config = f"({N} elm)"
        self.marker = marker
        self.array_shape = N

        # Power and noise power
        if dbm:
            self.power_dbm = power
            self.noise_power_dbm = noise_power
        else:
            self.power = power
            self.noise_power = noise_power

    N = num_antennas = property(lambda self: self.coordinates.shape[0])

    def grid_coord(self, N: ArrayLike | int, spacing: float = 0.5):
        """Set the coordinates of the antennas in a grid pattern centering at the origin.

        Args:
            N (int | Iterable): Number of antennas in the grid.
                The dimension of `N` will be padded to 3,
                and each element corresponds to the number of antennas in [x, z, y] respectively.
            spacing: Spacing between the antennas. Default is 0.5.

        Returns:
            numpy.ndarray: Coordinates of the antennas in the grid pattern.
        """
        if isinstance(N, int):
            N = [N]

        N = np.concatenate((N, np.ones(3 - len(N))), axis=0).astype(int)

        x = np.arange(N[0]) * spacing
        z = np.arange(N[1]) * spacing
        y = np.arange(N[2]) * spacing

        X, Y, Z = np.meshgrid(x, y, z)
        coordinates = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        # center the coordinates around the origin
        coordinates -= coordinates.mean(axis=0)
        return coordinates

    @classmethod
    def ula(
        cls,
        N,
        array_center=[0, 0, 0],
        ax="x",
        spacing=0.5,
        **kwargs,
    ):
        """Creates a half-wavelength spaced, uniform linear array along the desired axis.

        Args:
            N: Number of antennas in the array.
            array_center: Coordinates of the center of the array. Default is [0, 0, 0].
            ax: Axis along which the array is to be created.
                Takes value 'x', 'y' or 'z'. Default is 'x'.
            spacing: Spacing between the antennas. Default is 0.5.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            AntennaArray: A uniform linear array instance.

        Raises:
            ValueError: If ax is not 'x', 'y' or 'z'.
        """
        if ax == "x":
            coordinates = np.array([np.arange(N), np.zeros(N), np.zeros(N)]).T
        elif ax == "y":
            coordinates = np.array([np.zeros(N), np.arange(N), np.zeros(N)]).T
        elif ax == "z":
            coordinates = np.array([np.zeros(N), np.zeros(N), np.arange(N)]).T
        else:
            raise ValueError("axis must be 'x', 'y' or 'z'")

        ula = cls(N, coordinates * spacing, **kwargs)
        ula.array_center = array_center

        config_map = {"x": f"({N}x1x1)", "y": f"(1x{N}x1)", "z": f"(1x1x{N})"}
        ula._config = config_map[ax]

        return ula

    @classmethod
    def upa(
        cls,
        N: Iterable | int,
        array_center=(0, 0, 0),
        plane="xz",
        spacing=0.5,
        **kwargs,
    ):
        """Creates a half-wavelength spaced, uniform planar array in the desired plane.

        Args:
            N: Number of rows and columns in the array.
            array_center: Coordinates of the center of the array. Default is [0, 0, 0].
            plane: Plane in which the array is to be created or the axis orthogonal to the plane.
                Takes value 'xy', 'yz' or 'xz'. Default is 'xz'.
            spacing: Spacing between the antennas. Default is 0.5.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            AntennaArray: A uniform planar array instance.

        Raises:
            ValueError: If plane is not 'xy', 'yz', or 'xz'.
        """
        if isinstance(N, int):
            return cls.ula(
                N, array_center=array_center, ax=plane, spacing=spacing, **kwargs
            )

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
            "xy": f"({num_rows}x{num_cols}x1)",
            "yz": f"(1x{num_rows}x{num_cols})",
            "xz": f"({num_rows}x1x{num_cols})",
        }
        upa._config = config_map[plane]
        return upa

    # =============================
    # ======= Properties ==========
    # =============================

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name + " " + self._config

    def __len__(self):
        return self.num_antennas
    

    # safe power properties for numerical stability
    @property
    def _noise_power(self):
        return self.noise_power if self.noise_power > 0 else np.finfo(float).tiny

    power_dbm = property(lambda self: 10 * np.log10(self.power))
    noise_power_dbm = property(lambda self: 10 * np.log10(self._noise_power))

    @power_dbm.setter
    def power_dbm(self, power_dbm):
        self.power = 10 ** (power_dbm / 10)

    @noise_power_dbm.setter
    def noise_power_dbm(self, noise_power_dbm):
        self.noise_power = 10 ** (noise_power_dbm / 10)

    amp = property(lambda self: np.abs(self.weights))
    phase = property(lambda self: np.angle(self.weights))
    array_center = property(lambda self: np.mean(self.coordinates, axis=0))
    location = property(lambda self: np.mean(self.coordinates, axis=0))

    @array_center.setter
    def array_center(self, center):
        """Set the center of the array."""
        delta_center = center - self.array_center
        self.coordinates += delta_center

    @location.setter
    def location(self, location):
        """Set the location of the array."""
        delta_location = location - self.location
        self.coordinates += delta_location

    coord_x = property(lambda self: self.coordinates[:, 0])
    coord_y = property(lambda self: self.coordinates[:, 1])
    coord_z = property(lambda self: self.coordinates[:, 2])

    @property
    def diameter(self):
        """Returns the diameter of the array."""
        Dx = np.max(self.coordinates[:, 0]) - np.min(self.coordinates[:, 0])
        Dy = np.max(self.coordinates[:, 1]) - np.min(self.coordinates[:, 1])
        Dz = np.max(self.coordinates[:, 2]) - np.min(self.coordinates[:, 2])
        return np.sqrt(Dx**2 + Dy**2 + Dz**2)

    @diameter.setter
    def diameter(self, diameter):
        """Set the diameter of the array by scaling the coordinates."""
        scale = diameter / self.diameter
        self.coordinates *= scale

    @classmethod
    def from_file(cls, filename):
        """Load an antenna array from a file.

        Args:
            filename: Name of the file to load the array from.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError

    def to_file(self, filename):
        """Save the array to a file.

        Args:
            filename: Name of the file to save the array to.
        """
        np.save(filename, [self.coordinates, self.weights, self.marker])

    def normalize_weights(self, norm=1):
        """Normalize the weights of the antennas to have unit norm.

        Args:
            norm: Target norm for the weights. Default is 1.
        """
        if LA.norm(self.weights) != 0:
            self.weights = self.weights * norm / LA.norm(self.weights)

    def set_weights(self, weights: ArrayLike):
        """Set the weights of the antennas.

        Args:
            weights: Weights of the antennas. If a scalar is given, all
                antennas are set to the same weight. If an array is given, the
                weights are set to the given array. The shape of the array must
                match the number of antennas in the array.

        Raises:
            ValueError: If the length of weights does not match the number of antennas.
        """
        if np.isscalar(weights):
            self.weights = weights * np.ones(self.num_antennas)
        else:
            weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
            if len(weights) != self.num_antennas:
                raise ValueError(
                    "The length of weights must match the number of antennas"
                )
            self.weights = np.asarray(weights).reshape(-1)

    def get_weights(self, coordinates=None):
        """Get the weights of the antennas.

        Args:
            coordinates: Coordinates of the antennas whose weight is to be retrieved. If not
                given, the weights of all antennas are returned.

        Returns:
            numpy.ndarray: Array of weights.

        Raises:
            ValueError: If no matching coordinates are found.
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

        Args:
            coordinates: Coordinates of the antennas to be matched. The shape of the array must be (num_antennas, 3).

        Returns:
            numpy.ndarray: Indices of matching coordinates.
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

        Args:
            coordinates: Coordinates of the antennas to be added. The shape of the array must be (num_antennas, 3).
        """
        self.coordinates = np.concatenate((self.coordinates, coordinates))
        self.weights = np.concatenate((self.weights, np.ones(coordinates.shape[0])))

    def remove_elements(self, coordinates=None, indices=None):
        """Remove antennas from the array by coordinates or indices.

        Args:
            coordinates: Coordinates of the antennas to be removed. The shape of the array must be (num_antennas, 3).
            indices: Indices of the antennas to be removed.

        Raises:
            ValueError: If neither coordinates nor indices are provided.
        """

        def _remove_elements_by_coord(self, coordinates):
            indices = self._match_coordinates(coordinates)
            self.coordinates = np.delete(self.coordinates, indices, axis=0)
            self.weights = np.delete(self.weights, indices, axis=0)

        def _remove_elements_by_idx(self, indices):
            self.coordinates = np.delete(self.coordinates, indices, axis=0)
            self.weights = np.delete(self.weights, indices, axis=0)

        if coordinates is not None:
            _remove_elements_by_coord(coordinates)
        elif indices is not None:
            _remove_elements_by_idx(indices)
        else:
            raise ValueError("Either coordinates or indices must be given")

    def rotate(self, axis: str | float, angle: float, use_degrees=True):
        """Rotate the array counterclockwise with the given axis pointing out of the plane.

        Args:
            axis (`str` | `array_like`): Axis of rotation. Can be 'x', 'y', 'z' or a 3-element vector.
            angle (`array_like`): Angle of rotation in degrees or radians.
            use_degrees (`bool`): If True, the angles are in degrees. Default is True.
        """
        if use_degrees:
            angle = np.radians(angle)

        if isinstance(axis, str):
            axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            axis = axis_map[axis]

        rot_matrix = rotation_matrix(axis, angle)
        array_center = self.array_center

        # center the array at the origin
        self.array_center = [0, 0, 0]
        # Apply rotation matrix to all coordinates
        self.coordinates = np.dot(self.coordinates, rot_matrix.T)
        # translate the array back to its original position
        self.array_center = array_center

    ############################
    # Get AntennaArray Properties
    ############################

    def get_array_response(self, az=0, el=0, use_degrees=False):
        """Returns the array response vector at a given azimuth and elevation.

        This response is simply the phase shifts experienced by the elements
        on an incoming wavefront from the given direction, normalized to the first
        element in the array.

        Args:
            az: Azimuth angle in radians.
            el: Elevation angle in radians.
            grid: If True, then the array response is calculated for all combinations of
                azimuth and elevation angles.
                Otherwise, the array response is calculated for the given azimuth and elevation
                pair. (their dimensions must match)
            use_degrees: If True, az and el are in degrees, otherwise in radians.

        Returns:
            numpy.ndarray: The array response vector up to 3 dimensions. The shape of the array is
            (len(az), len(el), len(coordinates)) and is squeezed if az and/or el are scalars.
        """
        
        dx = (self.coord_x - self.coord_x[0]).reshape(1, -1)
        dy = (self.coord_y - self.coord_y[0]).reshape(1, -1)
        dz = (self.coord_z - self.coord_z[0]).reshape(1, -1)

        az = np.asarray(az).reshape(-1, 1)
        el = np.asarray(el).reshape(-1, 1)
        if use_degrees:
            az = np.deg2rad(az)
            el = np.deg2rad(el)

        array_response = np.exp(
            (1j * 2 * np.pi)
            * (
                dx * np.sin(az) * np.cos(el)
                + dy * np.cos(az) * np.cos(el)
                + dz * np.sin(el)
            )
        )
        # array_response = np.squeeze(array_response, axis=0)
        # if self.num_antennas == 1:
        #     array_response = array_response.reshape(-1, 1)

        return array_response

    def get_array_gain(self, az, el, db=True, use_degrees=True):
        """Returns the array gain at a given azimuth and elevation.

        Args:
            az: Azimuth angle.
            el: Elevation angle.
            db: If True, the gain is returned in dB. Default is True.
            use_degrees: If True, az and el are in degrees, otherwise in radians. Default is True.

        Returns:
            numpy.ndarray: The array gain at the given azimuth and elevation
                with shape (len(az), len(el)).
        """
        if use_degrees:
            az = np.deg2rad(az)
            el = np.deg2rad(el)

        array_response = self.get_array_response(az, el)
        # multiply gain by the weights at the last dimension
        gain = (array_response @ self.weights.reshape(-1, 1)) ** 2
        gain = np.asarray(np.squeeze(gain))
        mag = np.abs(gain)
        # phase = np.angle(gain)
        # print(gain)
        if db:
            return 10 * log10(mag + np.finfo(float).tiny)
        return mag

    get_gain = get_array_gain

    ############################
    # Plotting
    ############################

    def plot_array(self, plane="xy", ax=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plot array in 2D projection.

        Args:
            plane (`str`): Plane to plot in. Can be 'xy', 'yz' or 'xz'.
            ax (`matplotlib.axes.Axes`): Matplotlib axes to plot on. If None, creates a new figure.
            **kwargs: Additional arguments to pass to the plotting function.

        Returns:
            tuple: Figure and axes objects.
        """

        return plot_arrays(self, plane=plane, ax=ax, **kwargs)

    def plot_array_3d(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plot the antenna array in 3D.

        Args:
            **kwargs: Additional arguments passed to scatter.

        Returns:
            tuple: Figure and axes objects.
        """
        return plot_arrays_3d(self, **kwargs)

    def plot_gain(
        self,
        weights: ArrayLike = None,
        axis: str = "az",
        angle: float = 0,
        angle_range: ArrayLike = np.linspace(-89, 89, 356),
        use_degrees: bool = True,
        db: bool = True,
        polar: bool = True,
        ax: Axes = None,
        **kwargs,
    ):
        """Plot the array pattern at a given elevation or azimuth.

        Args:
            weights (ArrayLike, optional): Weights of the antennas. If not given,
                the weights of the array are used.
            axis (str): Axis along which the gain is to be plotted.
                Takes value 'el' (elevation) or 'az' (azimuth). Default is 'az'.
            angle (float): Angle at which the gain is to be plotted along the given axis.
            angle_range (ArrayLike): Range of angles at which the gain is to be plotted.
            use_degrees (bool): If True, the angles are in degrees. Default is True.
            db (bool): If True, the gain is plotted in dB. Default is True.
            polar (bool): If True, a polar plot is created. Default is True.
            ax (matplotlib.axes.Axes, optional): Axes on which to plot the gain.

        Returns:
            matplotlib.axes.Axes: The axes object with the plot.
        """

        if weights is not None:
            orig_weights = self.get_weights()
            self.set_weights(weights)

        if not use_degrees:
            angle = np.rad2deg(angle)
            angle_range = np.rad2deg(angle_range)

        if axis == "az":
            el = np.asarray(angle)
            az = np.asarray(angle_range)
        elif axis == "el":
            az = np.asarray(angle)
            el = np.asarray(angle_range)
        else:
            raise ValueError("axis must be 'az' or 'el'")

        az = np.deg2rad(az)
        el = np.deg2rad(el)

        # vectorized version
        gain = self.get_array_gain(az, el, db=db, use_degrees=False)

        if ax is None:
            if polar:
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            else:
                fig, ax = plt.subplots()

        # check if ax is polar for given axis
        polar = ax.name == "polar" if polar else False
        if polar:
            ax.plot(angle_range * np.pi / 180, gain, **kwargs)
            # ax.set_theta_zero_location("E")
            ax.set_theta_zero_location("N")
            ax.set_thetamin(min(angle_range))
            ax.set_thetamax(max(angle_range))
            ax.set_ylabel("Gain (dB)")
            ax.set_xlabel("Azimuth (deg)" if axis == "el" else "Elevation (deg)")
            ax.set_theta_direction(-1)
        else:
            ax.plot(angle_range, gain, **kwargs)
            # ax.set_ylim(-(max(array_response)), max(array_response) + 10)
            ax.set_xlabel("Azimuth (deg)" if axis == "el" else "Elevation (deg)")
            ax.set_ylabel("Gain (dB)")

        title = f"{axis} = {angle} deg, max gain = {np.max(np.abs(gain)):.2f} dB"

        ax.set_title(title)
        ax.grid(True)
        if weights is not None:
            self.set_weights(orig_weights)
        if ax is None:
            plt.tight_layout()
            plt.show()

        if axis == "el":
            if polar:
                ax.set_theta_zero_location("W")

        return ax

    def plot_gain_3d(
        self,
        weights=None,
        az=np.linspace(-180, 180, 360),
        el=np.linspace(-90, 90, 180),
        ax=None,
        max_gain=None,
        min_gain=-10,
        polar=True,
        use_degrees=True,
        dB=True,
        **kwargs,
    ):
        """Plot 3D gain pattern of the antenna array.

        Args:
            weights: Weights to use for the plot. If None, the current weights are used.
            az: Azimuth angles to evaluate gain at.
            el: Elevation angles to evaluate gain at.
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            max_gain: Maximum gain value for color scaling. If None, uses the maximum actual gain.
            min_gain: Minimum gain value for color scaling.
            polar: If True, plots in polar coordinates, otherwise in Cartesian.
            use_degrees: If True, az and el are in degrees, otherwise in radians.
            dB: If True, plots gain in dB scale.
            **kwargs: Additional arguments passed to plot_surface.

        Returns:
            matplotlib.axes.Axes: The axes object with the plot.
        """
        if weights is not None:
            orig_weights = self.get_weights()
            self.set_weights(weights)
        if use_degrees:
            az = np.deg2rad(az)
            el = np.deg2rad(el)
        
        AZ, EL = np.meshgrid(az, el)

        gain = self.get_array_gain(az, el, db=dB, use_degrees=False)

        if max_gain is None:
            max_gain = np.max(gain)
        if min_gain is None:
            min_gain = np.min(gain)
        gain = np.clip(gain, min_gain, max_gain).T
        gain -= min_gain

        figsize = kwargs.pop("figsize", (8, 8))
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={"projection": "3d"},
            )

        norm = plt.Normalize(min_gain, max_gain)
        m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
        m.set_array(gain + min_gain)
        colors = m.to_rgba(gain + min_gain)

        if polar:
            X = gain * np.sin(AZ) * np.cos(EL)
            Y = gain * np.cos(AZ) * np.cos(EL)
            Z = gain * np.sin(EL)

            fig = ax.get_figure()
            fig.colorbar(m, ax=ax, shrink=0.5)
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=colors,
                rstride=kwargs.pop("rstride", 2),
                cstride=kwargs.pop("cstride", 2),
                linewidth=kwargs.pop("linewidth", 0),
                **kwargs,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            # ax.view_init(elev=30, azim=30)

        else:
            ax.plot_surface(
                AZ,
                EL,
                gain,
                cmap="magma",
                facecolors=colors,
                # linewidth=1,
                **kwargs,
            )
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Elevation (deg)")
            ax.set_zlabel("Gain (dB)")

        if ax is None:
            plt.tight_layout()
            plt.show()

        title = f"Max Gain: {np.max(np.abs(gain)):.2f} dB"
        ax.set_title(title)
        if weights is not None:
            self.set_weights(orig_weights)
        return ax

    # def plot_array_3d(self, **kwargs):
    #     """Plot the antenna array in 3D.

    #     Args:
    #         **kwargs: Additional arguments passed to scatter.

    #     Returns:
    #         matplotlib.axes.Axes: The axes object with the plot.
    #     """
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(
    #         self.coordinates[:, 0],
    #         self.coordinates[:, 1],
    #         self.coordinates[:, 2],
    #         marker=self.marker,
    #     )
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     plt.tight_layout()
    #     plt.show()

    # def plot_array(self, plane="xy", ax=None):
    #     """Plot the array in 2D projection.

    #     Args:
    #         plane: Plane in which the array is to be projected.
    #             Takes value 'xy', 'yz' or 'xz'. Default is 'xy'.
    #         ax: Matplotlib axes to plot on. If None, creates a new figure.

    #     Returns:
    #         matplotlib.axes.Axes: The axes object with the plot.

    #     Raises:
    #         ValueError: If plane is not 'xy', 'yz', or 'xz'.
    #     """
    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     if plane == "xy":
    #         ax.scatter(
    #             self.coordinates[:, 0], self.coordinates[:, 1], marker=self.marker
    #         )
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y")
    #     elif plane == "yz":
    #         ax.scatter(
    #             self.coordinates[:, 1], self.coordinates[:, 2], marker=self.marker
    #         )
    #         ax.set_xlabel("y")
    #         ax.set_ylabel("z")
    #     elif plane == "xz":
    #         ax.scatter(
    #             self.coordinates[:, 0], self.coordinates[:, 2], marker=self.marker
    #         )
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("z")
    #     else:
    #         raise ValueError("plane must be 'xy', 'yz' or 'xz'")
    #     ax.grid(True)
    #     ax.set_title(r"AntennaArray Projection in {}-plane".format(plane))

    #     if ax is None:
    #         plt.show()
    #     return ax
