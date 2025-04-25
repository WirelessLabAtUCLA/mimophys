from collections.abc import Iterable
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import log2, log10

from .channels import Channel
from .devices.antenna_array import AntennaArray


class Network:
    """Network class.

    Attributes
    ----------
        name (str): Network name.
        links (list): List of links in the network.
        nodes (tuple): Tuple of nodes in the network.
        connections (dict): Dictionary of connections in the network.
        ng (dict): Node group. Dictionary of List[nodes] in the network.
        lg (dict): Link group. Dictionary of List[links] in the network.
        loi (list): List of links of interest in the network.
        noi (list): List of nodes of interest in the network.
    """

    def __init__(self, name="Network", *args, **kwargs):
        self.name = name
        self.links: Dict[str, Channel] = {}
        self.connections: Dict[AntennaArray, Dict[str, List[Channel]]] = {}
        self.lg: Dict[str, List[Channel]] = {}
        self.ng: Dict[str, List[AntennaArray]] = {}
        self.loi: List[Channel] = []
        self.noi: List[AntennaArray] = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    # ===================================================================
    # Links and Nodes
    # ===================================================================

    @property
    def nodes(self) -> Dict[str, AntennaArray]:
        return {node.name: node for node in self.connections.keys()}

    @nodes.setter
    def nodes(self, _):
        raise AttributeError("Cannot set nodes directly. Use add_nodes() instead.")

    n = nodes
    l = property(lambda self: self.links)  # noqa: E741
    topology = property(lambda self: self.connections)
    link_groups = property(lambda self: self.lg)
    node_groups = property(lambda self: self.ng)

    def _add_node(self, node: AntennaArray):
        """Add a node to the network."""
        if node not in self.connections:
            self.connections[node] = {"dl": [], "ul": []}

    def add_nodes(self, nodes: Iterable[AntennaArray]):
        """Add nodes to the network."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self._add_node(node)
        else:
            self._add_node(nodes)

    def _add_link(self, link: Channel):
        """Add a link to the network."""
        # link.name = f'{len(self.links)}_' + link.name
        if link.name not in self.links and link not in self.links.values():
            self.links[link.name] = link
            self.add_nodes(link.tx)
            self.connections[link.tx]["dl"].append(link)
            self.add_nodes(link.rx)
            self.connections[link.rx]["ul"].append(link)

    def add_links(self, links: Iterable[Channel]):
        """Add links to the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._add_link(link)
        else:
            self._add_link(links)

    def _remove_node(self, node: AntennaArray):
        """Remove a node and all links associated with it from the network."""
        if node in self.connections:
            for link in self.connections[node]["dl"]:
                # the node is the tx; remove ul from link.rx
                self.links.pop(link, None)
                self.connections[link.rx]["ul"].remove(link)
            for link in self.connections[node]["ul"]:
                # the node is the rx; remove dl from link.tx
                self.links.pop(link, None)
                self.connections[link.tx]["dl"].remove(link)

    def remove_nodes(self, nodes):
        """Remove nodes from the network."""
        if nodes.__iter__:
            for node in nodes:
                self._remove_node(node)
        else:
            self._remove_node(nodes)

    def _remove_link(self, link: Channel | str):
        """Remove a link from the network."""
        if isinstance(link, str):
            link = self.links[link]
        self.links.pop(link.name, None)
        self.connections[link.tx]["dl"].remove(link)
        self.connections[link.rx]["ul"].remove(link)

    def remove_links(self, links):
        """Remove links from the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._remove_link(link)
        else:
            self._remove_link(links)

    def realize(self):
        """Realize the network."""
        for _, link in self.links.items():
            link.realize()

    def clear_weights(self):
        """Clear the weights of all nodes in the network."""
        for node in self.nodes.values():
            node.set_weights(1)

    def move_node(self, node: str | AntennaArray, location):
        """Move a node to a new location.

        Parameters
        ----------
        node : str or AntennaArray
            Node to move.
        location : array_like
            New location of the node."""
        if isinstance(node, str):
            node = self.nodes[node]
        node.location = np.asarray(location)
        for link in self.connections[node]["dl"]:
            link.realize()
        for link in self.connections[node]["ul"]:
            link.realize()

    # ===================================================================
    # Link measurement methods wrapper
    # ===================================================================

    def rx_power(self, link: Optional[Channel | str] = None) -> float:
        if link is None:
            return {lk: self.rx_power(lk) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.rx_power(lk) for lk in link}
        return link.rx_power

    def gain(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the beamforming gain of the link in dB."""
        if link is None:
            return {lk: self.gain(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.snr(lk, db) for lk in link}
        return link.bf_gain_db if db else link.bf_gain

    bf_gain = gain

    def signal_power(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the beamforming gain of the link in dB."""
        if link is None:
            return {lk: self.signal_power(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.snr(lk, db) for lk in link}
        return link.signal_power_dbm if db else link.signal_power

    def bf_noise_power(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the noise power after beamforming in dBm."""
        if link is None:
            return {lk: self.bf_noise_power(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.snr(lk, db=db) for lk in link}
        return link.bf_noise_power_dbm if db else link.rx._noise_power

    def snr(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the signal-to-noise ratio (SNR) of the link."""
        if link is None:
            return {lk: self.snr(lk, db) for lk in self.links.values()}
        if isinstance(link, Iterable):
            return [self.snr(lk, db) for lk in link]
        if isinstance(link, str):
            link = self.links[link]
        return link.snr_db if db else link.snr

    def snr_upper_bound(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the SNR upper bound of the link."""
        if link is None:
            return {lk: self.snr_upper_bound(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return [self.snr_upper_bound(lk, db) for lk in link]
        return link.snr_upper_bound_db if db else link.snr_upper_bound

    # ===================================================================
    # Network measurement methods
    # ===================================================================
    def interference(self, link=None, db=True) -> float:
        """Get the interference of the link."""
        # interference is the sum of bf gains of all other ul links of the rx
        if link is None:
            return {lk: self.interference(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        interference = 0
        for ul in self.connections[link.rx]["ul"]:
            if ul != link:
                interference += self.signal_power(ul, db=False)

        return 10 * log10(interference + np.finfo(float).tiny) if db else interference

    def inr(self, link=None, db=True) -> float:
        """Get the interference-to-noise ratio (INR) of the link in dB."""
        if link is None:
            return {lk: self.inr(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        inr = self.interference(link, db=False) / self.bf_noise_power(link, db=False)
        return 10 * log10(inr + np.finfo(float).tiny) if db else inr

    def sinr(self, link=None, db=True) -> float:
        """Get the signal-to-interference-plus-noise ratio (SINR) of the link in dB."""
        if link is None:
            return {lk: self.sinr(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.sinr(lk, db) for lk in link}
        sinr = self.signal_power(link, db=False) / (
            self.interference(link, db=False) + self.bf_noise_power(link, db=False)
        )
        return 10 * log10(sinr + np.finfo(float).tiny) if db else sinr

    def spectral_efﬁciency(self, link: Optional[Channel | str] = None) -> float:
        """Get the spectral efﬁciency of the link in bps/Hz."""
        if link is None:
            return {lk: self.spectral_efﬁciency(lk) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {lk: self.spectral_efﬁciency(lk) for lk in link}
        return float(log2(1 + self.sinr(link, db=False)))

    se = spectral_efficiency

    def inr_upper_bound(self, link: Optional[Channel | str] = None, db=True) -> float:
        """Get the INR upper bound of the link. See Eq. (9) in LoneSTAR"""
        if link is None:
            return {lk: self.inr_upper_bound(lk, db) for lk in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        sig_pow_nb = 0
        for ul in self.connections[link.rx]["ul"]:
            if ul != link:
                sig_pow_nb += ul.rx_power * ul.tx.N * ul.rx.N
        inr_ub = sig_pow_nb / link.rx._noise_power
        return 10 * log10(inr_ub + np.finfo(float).tiny) if db else inr_ub

    # ===================================================================
    # Plotting methods
    # ===================================================================
    def plot(self, labels=False, plane="xy", ax=None, **kwargs):
        """Plot the network."""
        coord_idx = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        if ax is None:
            _, ax = plt.subplots(**kwargs)
        for node, connection in self.connections.items():
            # plot nodes
            node_loc = node.location[coord_idx]
            # ax.scatter(*node_loc[coord_idx], 'o', label=node.name)
            style = "b" if (node in self.noi) else "k"
            ax.scatter(*node_loc, s=70, facecolors=style, label=node.name)
            if labels:
                ax.annotate(node.name, node_loc)
            # plot downlink
            for link in connection["dl"]:
                dl_loc = link.rx.location[coord_idx]
                style = "c-" if (link in self.loi) else "k:"
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    style,
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=link.rx.name,
                )
                if labels:
                    offset = np.random.uniform(dl_loc - node_loc) * 0.1
                    ax.annotate(link.name, (dl_loc + node_loc) / 2 + offset)
        plt.xlabel(f"{plane[0]}-axis")
        plt.ylabel(f"{plane[1]}-axis")
        plt.title(f"{self.name}")
        if ax is None:
            plt.show()

    def plot_3d(self, ax=None, labels=False, **kwargs):
        """Plot the network in 3D."""
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)
        for node, connection in self.connections.items():
            # plot nodes
            node_loc = node.location
            style = "b" if (node in self.noi) else "k"
            ax.scatter(*node_loc, s=70, facecolors=style, label=node.name)
            if labels:
                ax.text(*node_loc, node.name)
            # plot downlink
            for link in connection["dl"]:
                dl_loc = link.rx.location
                style = "c-" if (link in self.loi) else "k:"
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    style,
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=link.rx.name,
                )
                if labels:
                    ax.text(*(dl_loc + node_loc) / 2, link.name)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"{self.name}")
        plt.tight_layout()
        if ax is None:
            plt.show()
        return fig, ax

    def plot_gain(
        self, ng=None, polar=True, axes=None, weights=None, ylim=-20, **kwargs
    ):
        """Plot the beam pattern of the controlled nodes."""
        if ng is None:
            nodes = list(self.connections.keys())
        else:
            nodes = self.ng[ng]
        num_plots = len(nodes)
        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5 * num_cols, 5 * num_rows)
        if axes is None:
            if polar:
                fig, axes = plt.subplots(
                    num_rows, num_cols, subplot_kw={"polar": True}, **kwargs
                )
            else:
                fig, axes = plt.subplots(num_rows, num_cols, **kwargs)
        for i, (node, ax) in enumerate(zip(nodes, np.ravel(axes))):
            if weights is not None:
                if len(weights) != num_plots:
                    raise ValueError(
                        "The number of weights must be the same as the number of nodes."
                    )
                node.plot_gain(ax=ax, weights=weights[i], polar=polar)
            else:
                node.plot_gain(ax=ax, polar=polar)
            title = ax.get_title()
            ax.set_title(f"{node.name}: {title}")
        if polar:
            for ax in np.ravel(axes):
                ax.set_ylim(bottom=ylim)
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
        if axes is None:
            plt.tight_layout()
            plt.show()

    def plot_arrays(self, plane: str = "xy", ax=None, **kwargs):
        """Plot the arrays in the network."""
        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        for node in self.connections.keys():
            node.plot(plane=plane, ax=ax, **kwargs)
        if ax is None:
            plt.tight_layout()

        return fig, ax
