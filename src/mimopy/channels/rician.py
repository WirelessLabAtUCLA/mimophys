import numpy as np

from ..devices import AntennaArray
from .awgn import Channel
from .los import LoSChannel
from .path_loss import PathLoss
from .rayleigh import RayleighChannel
from .spherical_wave import SphericalWaveChannel


class RicianChannel(Channel):
    """Rician channel class.

    Unique Attributes
    ----------
        K (float): Rician K-factor in dB.
        H_los (np.ndarray): Line-of-sight channel matrix.
        H_nlos (np.ndarray): Non-line-of-sight channel matrix.
    """

    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        K: float = 10,
        nearfield: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(tx, rx, path_loss, *args, **kwargs)
        self.K = 10 ** (K / 10)  # Convert K-factor to linear scale
        self.nearfield = nearfield
        if nearfield:
            self.los = SphericalWaveChannel(tx, rx, path_loss, seed=self.seed)
        else:
            self.los = LoSChannel(tx, rx, path_loss, seed=self.seed)
        self.nlos = RayleighChannel(tx, rx, path_loss, seed=self.seed)

    def generate_channels(self, n_channels=1, return_subchannels=False):
        """Generate multiple channel matrices with static LoS component."""
        H_los = self.los.realize().H
        H_nlos = self.nlos.generate_channels(n_channels)
        H = np.sqrt(self.K / (self.K + 1)) * H_los + np.sqrt(1 / (self.K + 1)) * H_nlos
        if return_subchannels:
            return H, H_los, H_nlos
        return H

    def realize(self):
        """Realize the channel."""
        self.los.realize()
        self.nlos.realize()

        self.channel_matrix = (
            np.sqrt(self.K / (self.K + 1)) * self.los.H
            + np.sqrt(1 / (self.K + 1)) * self.nlos.H
        )
        return self