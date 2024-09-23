import numpy as np

from ..devices import AntennaArray
from ..utils.geometry import relative_position
from .awgn import Channel
from .path_loss import PathLoss


class LoSChannel(Channel):
    """Line-of-sight channel class.

    Unique Attributes
    -----------------
    aoa, aod: float, optional
        AoA/AoD. If not specified, the angles are
        calculated based on the relative position of the transmitter and receiver.
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

    @property
    def aoa(self):
        _, az, el = relative_position(self.tx.array_center, self.rx.array_center)
        return az, el

    @aoa.setter
    def aoa(self, _):
        raise Warning("Use realize() to set the AoA/AoD, ignoring the input.")

    aod = aoa

    def generate_channels(self, az, el) -> np.ndarray:
        tx_response = self.tx.get_array_response(az, el)
        rx_response = self.rx.get_array_response(az + np.pi, el + np.pi)
        if len(tx_response.shape) == 1:
            tx_response = tx_response.reshape(1, -1)
        if len(rx_response.shape) == 1:
            rx_response = rx_response.reshape(1, -1)
        H = np.einsum("ij, ik->ijk", rx_response, tx_response.conj())
        return H

    def realize(self) -> "LoSChannel":
        """Realize the channel."""
        # TODO: warning if channel matrix not updated/realized
        _, az, el = relative_position(self.tx.array_center, self.rx.array_center)
        tx_response = self.tx.get_array_response(az, el)
        rx_response = self.rx.get_array_response(az + np.pi, el + np.pi)
        self.H = np.outer(rx_response, tx_response.conj())
        self.normalize_energy(self._energy)
        return self

# add alias with deprecat warning
class LoS(LoSChannel):
    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "LoS is deprecated, use LoSChannel instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)