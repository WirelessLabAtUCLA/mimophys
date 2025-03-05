import numpy as np

from .awgn import Channel


class MultipathChannel(Channel):
    """Generic multipath channel class with LoS and NLoS components."""

    def __init__(
        self,
        los_channel: Channel,
        nlos_channel: Channel,
        K: float = 10,
        *args,
        **kwargs,
    ):
        self.K = 10 ** (K / 10)  # Convert K-factor to linear scale
        self.los_factor = np.sqrt(self.K / (self.K + 1))
        self.nlos_factor = np.sqrt(1 / (self.K + 1))

        self.los = los_channel
        self.nlos = nlos_channel

        self.tx = self.los.tx
        self.rx = self.los.rx

        # check if tx and rx are the same across los and nlos
        if self.los.tx is not self.nlos.tx:
            raise ValueError(
                "The transmitter in los and nlos channels are not the same."
            )
        if self.los.rx is not self.nlos.rx:
            raise ValueError("The receiver in los and nlos channels are not the same.")

        super().__init__(
            los_channel.tx, los_channel.rx, los_channel.path_loss, *args, **kwargs
        )

    seed = property(lambda self: self.los.seed)

    @seed.setter
    def seed(self, value):
        self.los.seed = value
        self.nlos.seed = value

    def generate_channels(self, n_channels=1, return_subchannels=False):
        H_los = self.los.realize().H
        H_nlos = self.nlos.generate_channels(n_channels)
        H = self.los_factor * H_los + self.nlos_factor * H_nlos
        if return_subchannels:
            return H, H_los, H_nlos
        return H

    def realize(self):
        """Realize the channel."""
        self.los.realize()
        self.nlos.realize()

        self.channel_matrix = (
            self.los_factor * self.los.H + self.nlos_factor * self.nlos.H
        )
        return self
