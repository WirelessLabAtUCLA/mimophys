import numpy as np

from ..devices import AntennaArray
from .awgn import Channel
from .path_loss import PathLoss


class SphericalWaveChannel(Channel):
    """Spherical wave channel."""

    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        **kwargs,
    ):
        super().__init__(tx, rx, path_loss, **kwargs)

    def realize(self) -> "SphericalWaveChannel":
        """Realize the channel."""
        tc = self.tx.coordinates
        rc = self.rx.coordinates
        dx = tc[:, 0].reshape(1, -1) - rc[:, 0].reshape(-1, 1)
        dy = tc[:, 1].reshape(1, -1) - rc[:, 1].reshape(-1, 1)
        dz = tc[:, 2].reshape(1, -1) - rc[:, 2].reshape(-1, 1)
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        # get relative phase shift
        phase_shift = 2 * np.pi * d
        phase_shift = phase_shift - phase_shift[0, 0]
        self.channel_matrix = 1 / d * np.exp(-1j * phase_shift)
        self.normalize_energy(self._energy)
        return self