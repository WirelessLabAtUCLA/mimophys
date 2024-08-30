import numpy as np

from ..devices import AntennaArray
from .awgn import Channel
from .path_loss import PathLoss


class Rayleigh(Channel):
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

    def realize(self):
        """Realize the channel. Energy is used to adjusting the expectation of the channel"""
        np.random.seed(self.seed)
        energy = self._energy / self.tx.N / self.rx.N
        shape = (self.rx.N, self.tx.N)
        self.channel_matrix = np.sqrt(energy / 2) * (
            np.random.randn(*shape) + 1j * np.random.randn(*shape)
        )
        return self
