import numpy as np
from .awgn import Channel


class SphericalWave(Channel):
    """Spherical wave channel."""

    def __init__(self, tx, rx, name, **kwargs):
        super().__init__(tx, rx, name, **kwargs)

    def realize(self):
        """Realize the channel."""
        tc = self.tx.coordinates
        rc = self.rx.coordinates

        dx = tc[:, 0].reshape(-1, 1) - rc[:, 0].reshape(1, -1)
        dy = tc[:, 1].reshape(-1, 1) - rc[:, 1].reshape(1, -1)
        dz = tc[:, 2].reshape(-1, 1) - rc[:, 2].reshape(1, -1)
        # dx[tx, rx]
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # get relative phase shift
        phase_shift = 2 * np.pi * d
        self.channel_matrix = np.exp(1j * phase_shift).T.conj()
        return self