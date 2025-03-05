"""Sample code for a full-duplex network with a base station, 1 uplink UE, 1 downlink UE."""

import numpy as np

from mimophys import AntennaArray, Network
from mimophys.channels import LoSChannel, RicianChannel
from mimophys.channels.path_loss import ConstantLoss

# Number of transmit antennas at the base station
Nt = 8
# Number of receive antennas at the base station
Nr = 8
# Number of uplink antennas at the user equipment
Nul = 1
# Number of downlink antennas at the user equipment
Ndl = 1

# Coordinates of the base station antennas
bstx_coord = np.array([0, 20, 0])
bsrx_coord = np.array([0, -20, 0])

# Transmit power of the base station
bstx_power = 8
# Noise power at the base station receive antennas
bsrx_noise = 1

# Coordinates of the user equipment antennas
uetx_coord = np.array([5000, 5000, 0])
uerx_coord = np.array([5000, -5000, 0])

# Transmit power of the user equipment
uetx_power = 8
# Noise power at the user equipment receive antennas
uerx_noise = 1

# Create uniform linear arrays
tx = AntennaArray.ula(Nt, name="tx", array_center=bstx_coord, power=bstx_power)
rx = AntennaArray.ula(Nr, name="rx", array_center=bsrx_coord, noise_power=bsrx_noise)
uetx = AntennaArray.ula(Nul, name="uetx", array_center=uetx_coord, power=uetx_power)
uerx = AntennaArray.ula(
    Ndl, name="uerx", array_center=uerx_coord, noise_power=uerx_noise
)

# Define the uplink and downlink channel as a line-of-sight (LoS) channel
ul = LoSChannel(uetx, rx, name="ul")
dl = LoSChannel(tx, uerx, name="dl")

# Define the self-interference channel as a Rician channel with near-field LoS component
# and far-field isotropic scattering component. Set the path loss to a constant value of -20 dB
si = RicianChannel(
    tx, rx, K=10, nearfield=True, name="si", path_loss=ConstantLoss(-20)
)

# Create a network named "FD Network"
net = Network("FD Network")
# Add the defined links (downlink, uplink, and self-interference) to the network
net.add_links([dl, ul, si])

# Label the links of interest (LoI) in the network link group
net.lg["loi"] = [dl, ul]
# Label the base station nodes in the network node group
net.ng["bs"] = [tx, rx]
# Label the user equipment nodes in the network node group
net.ng["ue"] = [uetx, uerx]


# Realize the network (compute the channel matrices for all links in the network)
net.realize()

# Apply conjugate beamforming
tx.set_weights(dl.H.conj())
rx.set_weights(ul.H.conj())

print("SNR", net.snr([dl, ul]))
print("INR", net.inr(ul))
print("SINR", net.sinr([dl, ul]))
print("INR upper bound", net.inr_upper_bound(ul))
print("SNR upper bound", net.snr_upper_bound([dl, ul]))
print("uplink channel capacity", ul.capacity)
print("downlink channel capacity", dl.capacity)
