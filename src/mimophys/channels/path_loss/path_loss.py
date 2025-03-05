from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy import log10

if TYPE_CHECKING:
    from ..awgn import Channel


class PathLoss(ABC):
    @abstractmethod
    def received_power(self, channel: Channel):
        """Return the received power at the receiver."""
        pass


class NoLoss(PathLoss):
    def __init__(self):
        self.loss = 1
        self.loss_db = 0 

    def __str__(self):
        return "no_loss"
    
    def __repr__(self):
        return "no_loss"

    def received_power(self, channel: Channel):
        return channel.tx.power


class ConstantLoss(PathLoss):
    def __init__(self, loss: float, db: bool = True):
        if db:
            self.loss = 10 ** (loss / 10)
        else:
            self.loss = loss

    def __str__(self):
        return f"constant loss {self.loss:.2f} ({self.loss_db:.2f} dB)"
    
    def __repr__(self):
        return f"constant loss {self.loss:.2f} ({self.loss_db:.2f} dB)"

    @property
    def loss_db(self):
        return 10 * log10(self.loss)

    def received_power(self, channel: Channel):
        return channel.tx.power / self.loss