from collections import namedtuple
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..devices import AntennaArray
from .los import Channel
from .path_loss import PathLoss

ArrayLikeInt = Union[NDArray[np.int_], Sequence[int]]


class RayClusterChannel(Channel):
    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        seed: int = None,
        min_rays: int = 1,
        max_rays: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        rays_per_cluster: Optional[int | ArrayLikeInt] = None,
        cluster_angle_distrubution: str = "uniform",
        ray_angle_distribution: str = "laplace",
        ray_std: float = 0.1,
        aoa_bounds: int | ArrayLikeInt = ((-np.pi, np.pi), (-np.pi, np.pi)),
        aod_bounds: Optional[int | ArrayLikeInt] = None,
        use_degrees: bool = False,
        *args,
        **kwargs,
    ):
        """Ray Cluster Channel Model.

        Args:
            cluster_ray_count (Interable): Number of rays per cluster.
            cluster_angle_distrubution (str): Distribution of the cluster angles. Default is 'uniform'.
            ray_angle_distribution (str): Distribution of the ray angles. Default is 'laplace'.
                Available distributions are 'laplace' and 'normal'.
            ray_std (float): Standard deviation of the ray angles. Default is 0.1.
            aoa_bounds (ArrayLike): Bounds of the AoA angles. Default is ((-pi, pi), (-pi, pi)).
                The first dimension is for azimuth and the second dimension is for elevation.
            aod_bounds (ArrayLike, optional): Bounds of the AoD angles. Default is None.
                The first dimension is for azimuth and the second dimension is for elevation.
                If None, the same bounds as aoa_bounds will be used.
            use_degrees (bool): Whether to use degrees for the angles and `ray_std`. Default is False.
                If True, the angles will be converted to radians.
        """

        super().__init__(tx, rx, path_loss, seed, *args, **kwargs)
        self.min_rays = min_rays
        self.max_rays = max_rays if max_rays is not None else min_rays
        self.min_clusters = min_clusters if min_clusters is not None else min_rays
        self.max_clusters = max_clusters if max_clusters is not None else min_clusters

        # allocate rays
        if rays_per_cluster is not None:
            self.rays_per_cluster = np.array(rays_per_cluster)
        else:
            self.rays_per_cluster = self.set_rays()

        self.cluster_angle_distrubution = cluster_angle_distrubution
        self.ray_angle_distribution = ray_angle_distribution

        self.ray_std = ray_std
        self.aoa_bounds = aoa_bounds
        self.aod_bounds = aod_bounds if aod_bounds is not None else aoa_bounds
        if use_degrees:
            self.ray_std = np.deg2rad(self.ray_std)
            self.aoa_bounds = np.deg2rad(self.aoa_bounds)
            self.aod_bounds = np.deg2rad(self.aod_bounds)

    def n_rays(self):
        return int(self.rays_per_cluster.sum())

    def n_clusters(self):
        return self.rays_per_cluster.shape[0]

    def set_rays(self):
        self.rays_per_cluster = self.draw_rays(1)[0]

    def draw_rays(self, n_channels: int = 1) -> list[np.ndarray]:
        """Draw rays from the distribution. It's sorted by the number of rays.

        Args:
            n_channels (int): The number of channels to draw.

        Returns:
            list[np.ndarray]: A list of rays for each channel.
                Each element is a 1D array of shape (num_rays,).
        """
        n_rays = self.rng.integers(self.min_rays, self.max_rays + 1, size=n_channels)
        n_rays.sort()

        min_clusters = np.minimum(n_rays, self.min_clusters)
        max_clusters = np.minimum(n_rays, self.max_clusters)
        n_clusters = self.rng.integers(min_clusters, max_clusters + 1, size=n_channels)

        return [
            self.rng.multinomial(
                n_rays[i] - n_clusters[i], [1 / n_clusters[i]] * n_clusters[i]
            )
            + 1
            for i in range(n_channels)
        ]

    def generate_ray_angles(
        self,
        rays_per_cluster: list[ArrayLikeInt],
    ) -> np.ndarray:
        """
        Generate AoA and AoD of the rays based on cluster.
        Number of clusters will be randomly generated while n_rays is fixed.

        Args:

        Returns:
            np.ndarray: AoA and AoD of the rays with shape (num_channels, num_rays, 2)
                The last dimension is for azimuth and elevation, respectively.
        """
        n_channels = len(rays_per_cluster)
        n_clusters = np.array([len(rpc) for rpc in rays_per_cluster])
        rays_count = np.concatenate(rays_per_cluster)

        unique_cluster_aoas = self.rng.uniform(
            *np.array(self.aoa_bounds).T, (n_clusters.sum(), 2)
        )
        unique_cluster_aods = self.rng.uniform(
            *np.array(self.aod_bounds).T, (n_clusters.sum(), 2)
        )

        # repeat the cluster centers for each ray in the cluster
        # shape: (n_channels, total_num_rays, 2)
        cluster_aoa = np.repeat(unique_cluster_aoas, rays_count, axis=0).reshape(
            n_channels, -1, 2
        )
        cluster_aod = np.repeat(unique_cluster_aods, rays_count, axis=0).reshape(
            n_channels, -1, 2
        )

        # generate the angles of the rays based on the cluster centers
        aoa = self.rng.laplace(loc=cluster_aoa, scale=self.ray_std)
        aod = self.rng.laplace(loc=cluster_aod, scale=self.ray_std)
        return aoa, aod, unique_cluster_aoas, unique_cluster_aoas

    def generate_ray_gain(self, angles) -> np.ndarray:
        """Generate gain of the rays with complex Gaussian distribution.

        Args:
            angles (np.ndarray): Angles of the rays with shape (num_channels, num_rays)
        Returns:
            np.ndarray: Gain of the rays with shape (num_channels, num_rays)
        """
        # aoa and aod have the same shape, so we can use either one for gain shape
        # shape is (num_channels, total_num_rays)
        ray_gain = self.rng.normal(0, np.sqrt(1 / 2), (*angles.shape[:-1], 2))
        ray_gain = ray_gain.view(np.complex128).reshape(*angles.shape[:-1])
        return ray_gain

    def compute_channel_matrix(self, aoa, aod, gain) -> np.ndarray:
        """Generate channel matrix based on the generated angles for fixed n_rays.

        Args:
            aoa (np.ndarray): AoA of the rays with shape (num_channels, num_rays)
            aod (np.ndarray): AoD of the rays with shape (num_channels, num_rays)
            gain (np.ndarray): Gain of the rays with shape (num_channels, num_rays)

        Returns:
            np.ndarray: Channel matrix with shape (num_channels, tx.N, rx.N)
        """
        aoa_az, aoa_el = aoa[..., 0], aoa[..., 1]
        aod_az, aod_el = aod[..., 0], aod[..., 1]
        n_rays = aoa.shape[1]
        arx = self.rx.get_array_response(aoa_az, aoa_el)
        atx = self.tx.get_array_response(aod_az, aod_el)
        arx = arx.reshape(*aoa_az.shape, -1)
        atx = atx.reshape(*aod_az.shape, -1)
        H = np.einsum("bn,bnr,bnt->brt", gain, arx, atx.conj())
        H /= np.sqrt(n_rays)
        return H.squeeze()

    def realize(self):
        """Realize the channel.

        Returns:
            RayClusterChannel: The realized channel object.
        """
        # cluster_aoa, cluster_aod = self.generate_cluster_angles(1)
        self.set_rays()
        self.aoas, self.aods, self.cluster_aoa, self.cluster_aod = (
            self.generate_ray_angles(rays_per_cluster=[self.rays_per_cluster])
        )
        ray_gain = self.generate_ray_gain(self.aoas)
        self.H = self.compute_channel_matrix(self.aoas, self.aods, ray_gain)
        return self

    def generate_channels(self, n_channels=1, return_params=False):
        """Generate channel matrices.

        Args:
            n_channels (int): The number of channel matrices to generate.
            return_params (bool): Whether to return the parameters used to generate the channel.

        Returns:
            np.ndarray: Channel matrices with shape (num_channels, tx.N, rx.N)
            RayClusterParams: The parameters used to generate the channel.
                The parameters are a named tuple with the following fields:
                - n_rays (np.ndarray): The number of rays for each channel.
                - aoas (list[np.ndarray]): A list of AoA of the rays for each channel,
                    with shape (num_rays, 2) for each element.
                - aods (list[np.ndarray]): A list of AoD of the rays for each channel,
                    with shape (num_rays, 2) for each element.
                - ray_gains (list[np.ndarray]): A list of gain of the rays for each channel,
                    with shape (num_rays,) for each element.
        """
        n_channels = int(n_channels)
        # generate unique n_rays
        rays_per_cluster = self.draw_rays(n_channels)

        n_rays = np.array([rpc.sum() for rpc in rays_per_cluster])
        H = np.empty((n_channels, self.rx.N, self.tx.N), dtype=np.complex128)

        # split rays_per_cluster by the number of rays
        _, n_ray_counts = np.unique(n_rays, return_counts=True)

        if return_params:
            aoas, aods, ray_gains = [], [], []

        cts = np.append(0, np.cumsum(n_ray_counts))
        for i in range(len(cts) - 1):
            rpc = rays_per_cluster[cts[i] : cts[i + 1]]
            aoa, aod, cluster_aoa, cluster_aod = self.generate_ray_angles(rpc)
            ray_gain = self.generate_ray_gain(aoa)
            H[cts[i] : cts[i + 1]] = self.compute_channel_matrix(aoa, aod, ray_gain)
            if return_params:
                aoas.append(aoa)
                aods.append(aod)
                ray_gains.append(ray_gain)

        if return_params:
            # return a named tuple
            RayClusterParams = namedtuple(
                "RayClusterParams", ["n_rays", "aoas", "aods", "ray_gains"]
            )
            return H, RayClusterParams(n_rays, aoas, aods, ray_gains)

        return H


# %%
