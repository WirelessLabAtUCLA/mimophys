from abc import abstractmethod

import numpy as np
import numpy.linalg as LA
from numpy import log2, log10

from ..devices.antenna_array import AntennaArray
from .path_loss import PathLoss, get_path_loss


class Channel:
    """Base class for AWGN Channel.

     Attributes
    ----------
        name (str): Channel name.
        tx (AntennaArray): Transmit array.
        rx (AntennaArray): Receive array.
        num_antennas_tx (int): Number of transmit antennas.
        num_antennas_rx (int): Number of receive antennas.
        propagation_velocity (float): Propagation velocity in meters per second.
        carrier_frequency (float): Carrier frequency in Hertz.
        carrier_wavelength (float): Carrier wavelength in meters.
    """

    def __init__(
        self,
        tx: AntennaArray,
        rx: AntennaArray,
        path_loss: str | PathLoss = "no_loss",
        seed: int = None,
        # *args,
        # **kwargs,
    ):
        # use class name as default name
        self.name = self.__class__.__name__
        self.tx = tx
        self.rx = rx
        # energy of the channel matrix TO BE REALIZED
        self._energy = self.tx.N * self.rx.N
        self.channel_matrix = -np.ones((self.rx.N, self.tx.N), dtype=complex)
        self.seed = seed

        self._carrier_frequency = 1e9
        self._propagation_velocity = 299792458
        self._carrier_wavelength = self.propagation_velocity / self.carrier_frequency
        if isinstance(path_loss, str):
            self.path_loss = get_path_loss(path_loss)
        elif isinstance(path_loss, PathLoss):
            self.path_loss = path_loss
        else:
            raise ValueError("path_loss must be a string or PathLoss object.")

        # for kw, arg in kwargs.items():
        #     setattr(self, kw, arg)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.name} ({self.__class__.__name__})"

    seed = property(lambda self: self._seed)

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.rng = np.random.default_rng(self._seed)

    H = property(lambda self: self.channel_matrix)

    @H.setter
    def H(self, H):
        self.channel_matrix = H

    @property
    def energy(self):
        """Energy of the channel matrix."""
        return LA.norm(self.H, "fro") ** 2

    @energy.setter
    def energy(self, energy):
        self._energy = energy

    @property
    def nodes(self):
        return [self.tx, self.rx]

    def has_node(self, node):
        return node == self.tx or node == self.rx

    # ========================================================
    # Channel matrix
    # ========================================================

    @abstractmethod
    def generate_channels(self, n_channels=1):
        """Generate multiple channel matrices."""
        self.rng = np.random.default_rng(self._seed)

    @abstractmethod
    def realize(self):
        """Realize the channel."""
        self.rng = np.random.default_rng(self._seed)

    @staticmethod
    def normalize_channel(H, energy):
        """Normalize the channel energy."""
        H = np.sqrt(energy) * H / LA.norm(H, "fro")
        return H

    def normalize_energy(self, energy):
        """Normalize the channel energy."""
        if energy is not None:
            self.H = self.normalize_channel(self.H, energy)
        return self.H

    # ========================================================
    # Measurements
    # ========================================================
    @property
    def rx_power(self):
        """Received power in linear scale."""
        return self.path_loss.received_power(self)

    @property
    def bf_noise_power(self):
        """Noise power after beamforming combining in linear scale."""
        # w = self.rx.weights.flatten()
        # return float(LA.norm(w) ** 2 * self.rx.noise_power_lin)
        return float(self.rx._noise_power)

    @property
    def bf_noise_power_dbm(self) -> float:
        """Noise power after beamforming in dBm."""
        return 10 * log10(self.rx._noise_power + np.finfo(float).tiny)

    @property
    def bf_gain(self) -> float:
        """Normalized beamforming gain |wHf|^2 / Nt in linear scale."""
        f = self.tx.weights.reshape(-1, 1)
        w = self.rx.weights.reshape(-1, 1)
        return float(np.abs(w.T @ self.H @ f) ** 2 / (self.tx.N * LA.norm(w) ** 2))

    @property
    def bf_gain_db(self) -> float:
        """Normalized beamforming gain |wHf|^2 / Nt in dB."""
        return 10 * log10(self.bf_gain + np.finfo(float).tiny)

    gain = bf_gain
    gain_db = bf_gain_db

    @property
    def signal_power(self) -> float:
        """Signal power after beamforming in linear scale."""
        return self.rx_power * self.bf_gain

    @property
    def signal_power_dbm(self) -> float:
        """Normalized signal power after beamforming in dBm."""
        return 10 * log10(self.signal_power + np.finfo(float).tiny)

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio (SNR) in linear scale."""
        return float(self.rx_power * self.bf_gain / self.rx._noise_power)

    @property
    def snr_db(self) -> float:
        """Signal-to-noise ratio (SNR) in dB."""
        return 10 * log10(self.snr + np.finfo(float).tiny)

    @property
    def capacity(self) -> float:
        """Channel capacity in bps/Hz."""
        return log2(1 + self.snr_upper_bound)

    @property
    def gain_upper_bound(self) -> float:
        """return the gain upper bound based on MRC+MRT with line-of-sight channel"""
        return self.tx.N * self.rx.N

    @property
    def snr_upper_bound(self) -> float:
        """return the SNR upper bound based on MRC+MRT with line-of-sight channel"""
        return self.rx_power * self.tx.N * self.rx.N / self.rx._noise_power

    @property
    def snr_upper_bound_db(self) -> float:
        """return the SNR upper bound based on MRC+MRT with line-of-sight channel"""
        return 10 * log10(self.snr_upper_bound + np.finfo(float).tiny)

    # ========================================================
    # Skip Setters
    # ========================================================

    @signal_power.setter
    def signal_power(self, _):
        self._cant_be_set()

    @signal_power_dbm.setter
    def signal_power_dbm(self, _):
        self._cant_be_set()

    @bf_noise_power.setter
    def bf_noise_power(self, _):
        self._cant_be_set()

    @bf_noise_power_dbm.setter
    def bf_noise_power_dbm(self, _):
        self._cant_be_set()

    @snr.setter
    def snr(self, _):
        self._cant_be_set()

    @snr_db.setter
    def snr_db(self, _):
        self._cant_be_set()

    @capacity.setter
    def capacity(self, _):
        self._cant_be_set()

    @staticmethod
    def _cant_be_set():
        # raise warning
        raise Warning("This property can't be set, skipping...")

    # ========================================================
    # Physical properties
    # ========================================================
    @property
    def carrier_frequency(self):
        """Carrier frequency in Hertz.
        Also update carrier wavelength when set."""
        return self._carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, carrier_frequency):
        self._carrier_frequency = carrier_frequency
        self._carrier_wavelength = self.propagation_velocity / carrier_frequency

    @property
    def propagation_velocity(self):
        """Propagation velocity in meters per second.
        Also update carrier wavelength when set."""
        return self._propagation_velocity

    @propagation_velocity.setter
    def propagation_velocity(self, propagation_velocity):
        self._propagation_velocity = propagation_velocity
        self._carrier_wavelength = propagation_velocity / self.carrier_frequency

    @property
    def carrier_wavelength(self):
        """Carrier wavelength in meters.
        Also update carrier frequency when set."""
        return self._carrier_wavelength

    @carrier_wavelength.setter
    def carrier_wavelength(self, carrier_wavelength):
        self._carrier_wavelength = carrier_wavelength
        self._carrier_frequency = self.propagation_velocity / carrier_wavelength
