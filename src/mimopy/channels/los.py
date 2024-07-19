from .awgn import Channel
import numpy as np
from ..utils.geometry import relative_position


class LoS(Channel):
    """Line-of-sight channel class.

    Unique Attributes
    -----------------
    aoa, aod: float, optional
        AoA/AoD. If not specified, the angles are
        calculated based on the relative position of the transmitter and receiver.
    """

    def __init__(self, tx, rx, path_loss="no_loss", *args, **kwargs):
        super().__init__(tx, rx, path_loss, *args, **kwargs)

    @property
    def aoa(self):
        range, az, el = relative_position(self.tx.array_center, self.rx.array_center)
        return az, el

    @aoa.setter
    def aoa(self, _):
        raise Warning("Use realize() to set the AoA/AoD, ignoring the input.")

    aod = aoa

    def _compute_H(self, az, el):
        tx_response = self.tx.get_array_response(az, el)
        rx_response = self.rx.get_array_response(az + np.pi, el + np.pi)
        H = np.einsum("ij, ik->ijk", rx_response, tx_response.conj())
        return H

    def realize(self):
        """Realize the channel."""
        # TODO: warning if channel matrix not updated/realized
        range, az, el = relative_position(self.tx.array_center, self.rx.array_center)
        tx_response = self.tx.get_array_response(az, el)
        rx_response = self.rx.get_array_response(az + np.pi, el + np.pi)
        self.H = np.outer(rx_response, tx_response.conj())
        self.normalize_energy(self._energy)
        return self
