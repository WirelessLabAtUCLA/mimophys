# %%
"""TODO: Warning: The port from `beamgen` is not thoroughly tested."""

import numpy as np
import torch
from numpy.typing import ArrayLike

from ..devices import AntennaArray
from .los import Channel
from .path_loss import PathLoss


class RayClusterChannel(Channel):
    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        seed: int = 0,
        n_clusters: int = 1,
        n_rays: int | ArrayLike = 1,
        cluster_angle_distrubution: str = "uniform",
        ray_angle_distribution: str = "laplace",
        ray_std: float = 0.1,
        aoa_bounds: ArrayLike = ((-np.pi, np.pi), (-np.pi, np.pi)),
        aod_bounds: ArrayLike = ((-np.pi, np.pi), (-np.pi, np.pi)),
        use_degrees: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(tx, rx, path_loss, seed, *args, **kwargs)
        self.n_clusters = n_clusters
        self.cluster_angle_distrubution = cluster_angle_distrubution
        self.ray_angle_distribution = ray_angle_distribution
        self.ray_std = ray_std
        self.aoa_bounds = aoa_bounds
        self.aod_bounds = aod_bounds
        self._n_rays = -1
        self.n_rays = n_rays
        
        if use_degrees:
            self.aoa_bounds = np.deg2rad(self.aoa_bounds)
            self.aod_bounds = np.deg2rad(self.aod_bounds)

    n_rays = property(lambda self: self._n_rays)
    total_n_rays = property(lambda self: self.n_rays.sum())

    @n_rays.setter
    def n_rays(self, n_rays):
        if isinstance(n_rays, int):
            self._n_rays = np.full(self.n_clusters, n_rays)
        else:
            self._n_rays = np.array(n_rays)
            self.n_clusters = len(self._n_rays)

    def generate_cluster_angles(self, n_channels) -> np.ndarray:
        """Generate AoA and AoD of the cluster centers.

        Returns:
            np.ndarray: AoA and AoD of the clusters with shape (num_channels, num_clusters, 2)
                The last dimension is for azimuth and elevation, respectively.
        """
        rv = getattr(self.rng, self.cluster_angle_distrubution)
        cluster_aoa = rv(*np.array(self.aoa_bounds).T, (n_channels, self.n_clusters, 2))
        cluster_aod = rv(*np.array(self.aod_bounds).T, (n_channels, self.n_clusters, 2))
        return cluster_aoa, cluster_aod

    def generate_ray_angles(self, cluster_aoa, cluster_aod) -> np.ndarray:
        """Generate individual AoA and AoD of rays based on cluster.

        Parameters:
            distrubution (str): The distribution of the ray angles. Default is 'laplace'.
            torch (bool): If True, use PyTorch to generate the angles. Default is False.
        Returns:
            np.ndarray: AoA and AoD of the rays with shape (num_channels, num_rays)
        """
        rv = getattr(self.rng, self.ray_angle_distribution)
        aoa = rv(
            loc=cluster_aoa.repeat(self.n_rays, axis=1),
            scale=self.ray_std,
        )
        aod = rv(
            loc=cluster_aod.repeat(self.n_rays, axis=1),
            scale=self.ray_std,
        )
        return aoa, aod

    def generate_ray_gain(self, aoa) -> np.ndarray:
        """Generate gain of the rays with complex Gaussian distribution."""
        # aoa and aod have the same shape, so we can use either one for gain shape
        # shape is (num_channels, total_num_rays)
        ray_gain = self.rng.normal(0, np.sqrt(1 / 2), (*aoa.shape[:-1], 2))
        ray_gain = ray_gain.view(np.complex128).reshape(*aoa.shape[:-1])
        return ray_gain

    def generate_channel_matrix(self, aoa, aod, gain, use_torch=False) -> np.ndarray:
        """Generate channel matrix based on the generated angles.

        Parameters:
            use_torch (bool): If True, use PyTorch to generate the channel matrix. Default is False.
                Caution: Nor VRAM effecient!.
        Returns:
            np.ndarray: Channel matrix with shape (num_channels, tx.N, rx.N)
        """
        if use_torch:
            return self._torch_generate_channel_matrix(aoa, aod, gain)
        aoa_az, aoa_el = aoa[..., 0], aoa[..., 1]
        aod_az, aod_el = aod[..., 0], aod[..., 1]
        arx = self.rx.get_array_response(aoa_az, aoa_el, grid=False)
        atx = self.tx.get_array_response(aod_az, aod_el, grid=False)
        arx = arx.reshape(*aoa_az.shape, -1)
        atx = atx.reshape(*aod_az.shape, -1)
        H = np.einsum("bn,bnr,bnt->brt", gain, arx, atx.conj())
        H /= np.sqrt(self.total_n_rays)
        return H.squeeze()

    def _torch_generate_channel_matrix(self, aoa, aod, gain):
        aoa = torch.as_tensor(aoa, dtype=torch.float64, device=self.device)
        aod = torch.as_tensor(aod, dtype=torch.float64, device=self.device)
        aoa_az, aoa_el = aoa[..., 0], aoa[..., 1]
        aod_az, aod_el = aod[..., 0], aod[..., 1]
        arx = self.rx.get_array_response(
            aoa_az, aoa_el, torch_device=self.device, grid=False, return_tensor=True
        )
        atx = self.tx.get_array_response(
            aod_az, aod_el, torch_device=self.device, grid=False, return_tensor=True
        )
        arx = arx.reshape(*aoa.shape, -1)
        atx = atx.reshape(*aod.shape, -1)
        gain = torch.as_tensor(gain, dtype=torch.complex128, device=self.device)
        H = torch.einsum("bn,bnr,bnt->brt", gain, arx, atx.conj())
        H /= np.sqrt(self.total_n_rays).cpu().numpy()
        del arx, atx, gain
        torch.cuda.empty_cache()
        return H

    def generate_channels(self, n_channels=1, use_torch=False, return_params=False):
        n_channels = int(n_channels)
        cluster_aoa, cluster_aod = self.generate_cluster_angles(n_channels)
        aoa, aod = self.generate_ray_angles(cluster_aoa, cluster_aod)
        ray_gain = self.generate_ray_gain(aoa)
        H = self.generate_channel_matrix(aoa, aod, ray_gain, use_torch)
        if return_params:
            return H, cluster_aoa, cluster_aod, aoa, aod, ray_gain
        return H

    def realize(self):
        """Realize the channel."""
        cluster_aoa, cluster_aod = self.generate_cluster_angles(1)
        aoa, aod = self.generate_ray_angles(cluster_aoa, cluster_aod)
        ray_gain = self.generate_ray_gain(aoa)
        (self.cluster_aoa, self.cluster_aod) = (cluster_aoa, cluster_aod)
        (self.aoa, self.aod, self.ray_gain) = (aoa, aod, ray_gain)
        self.H = self.generate_channel_matrix(aoa, aod, ray_gain)
        return self

# %%
