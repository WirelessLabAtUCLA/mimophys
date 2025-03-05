import numpy as np

from ..devices import AntennaArray
from .awgn import Channel
from .path_loss import PathLoss


class RayleighChannel(Channel):
    """Rayleigh channel class.

    Unique Attributes
    -----------------
    seed: int, optional
        Seed for random number generator.
    """

    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        *args,
        **kwargs,
    ):
        super().__init__(tx, rx, path_loss, *args, **kwargs)

    def generate_channels(self, n_channels=1):
        # super().generate_channels(n_channels)
        shape = (self.rx.N, self.tx.N)
        energy = self._energy / self.tx.N / self.rx.N
        H = self.rng.normal(0, np.sqrt(energy / 2), (n_channels, *shape, 2))
        H = H.view(np.complex128).squeeze()
        return H

    def realize(self):
        """Realize the channel. Energy is used to adjusting the expectation of the channel"""
        super().realize()
        energy = self._energy / self.tx.N / self.rx.N
        shape = (self.rx.N, self.tx.N)
        # self.channel_matrix = np.sqrt(energy / 2) * (
        #     np.random.randn(*shape) + 1j * np.random.randn(*shape)
        # )
        self.channel_matrix = self.rng.normal(0, np.sqrt(energy / 2), (*shape, 2))
        self.channel_matrix = self.channel_matrix.view(np.complex128).squeeze()
        return self
