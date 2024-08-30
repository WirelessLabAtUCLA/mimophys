import numpy as np

from ..devices import AntennaArray
from .awgn import Channel
from .los import LoS
from .path_loss import PathLoss
from .rayleigh import Rayleigh
from .spherical_wave import SphericalWave


class Rician(Channel):
    """Rician channel class.

    Unique Attributes
    ----------
        K (float): Rician K-factor.
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
            self.los = SphericalWave(tx, rx, path_loss)
        else:
            self.los = LoS(tx, rx, path_loss)
        self.nlos = Rayleigh(tx, rx, path_loss, seed=self.seed)

    def realize(self):
        """Realize the channel."""
        self.los.realize()

        np.random.seed(self.seed)
        self.nlos.seed = self.seed
        self.nlos.realize()

        self.channel_matrix = (
            np.sqrt(self.K / (self.K + 1)) * self.los.H
            + np.sqrt(1 / (self.K + 1)) * self.nlos.H
        )
        return self
