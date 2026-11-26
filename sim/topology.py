"""P2P blockchain overlay topology and BIP-152 compact-block propagation delay model.

Builds a Bitcoin-calibrated degree-distributed 1,000-node network (and 50-node WAN topology
with log-normal link delays and packet loss) for computing block gossip propagation times.
"""

from __future__ import annotations
import numpy as np


class P2PNetworkTopology:
    """Undirected P2P overlay network with geographic link latencies and compact-block relay."""

    def __init__(
        self,
        num_nodes: int,
        rng: np.random.Generator,
        wan_mode: bool = False,
        mu_ln: float = 4.5,
        sigma_ln: float = 0.9,
        packet_loss: float = 0.005,
        backbone_bw_mbps: float = 100.0,
        edge_bw_mbps: float = 15.0,
    ) -> None:
        self.num_nodes = num_nodes
        self.wan_mode = wan_mode
        self.packet_loss = packet_loss
        self.rng = rng

        # Node roles: 20% backbone super-relays (100 Mbps), 80% edge miners (10-20 Mbps, mean 15 Mbps)
        n_backbone = max(4, int(0.20 * num_nodes))
        self.is_backbone = np.zeros(num_nodes, dtype=bool)
        self.is_backbone[:n_backbone] = True

        edge_lo = max(2.0, edge_bw_mbps - 5.0)
        edge_hi = edge_bw_mbps + 5.0
        self.node_bw_mbps = np.where(
            self.is_backbone,
            backbone_bw_mbps,
            rng.uniform(edge_lo, edge_hi, size=num_nodes),
        )

        # Regional cluster assignment (5 WAN regions in 50-node mode; 6 global regions in 1,000-node overlay)
        n_regions = 5 if wan_mode else 6
        self.region = rng.integers(0, n_regions, size=num_nodes)
        if wan_mode:
            # Heavy-tailed WAN delay: log-normal in ms converted to seconds
            hop_ms = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=num_nodes)
            self.base_hop_delay_s = np.clip(hop_ms / 1000.0, 0.015, 0.900)
        else:
            # SimBlock regional RTT tiering (~45-125 ms per hop across global pools)
            region_factor = np.array([0.045, 0.060, 0.075, 0.090, 0.105, 0.125])
            self.base_hop_delay_s = region_factor[self.region] * rng.uniform(0.85, 1.15, size=num_nodes)

        # Degree distribution (mean degree ~ 8-9 -> effective gossip depth ~ 3.2-3.6 hops)
        self.degrees = np.where(
            self.is_backbone,
            rng.integers(12, 18, size=num_nodes),
            rng.integers(6, 11, size=num_nodes),
        )
        mean_deg = float(self.degrees.mean())
        self.nominal_diameter_hops = max(2.4, float(np.log(num_nodes) / np.log(max(2.0, mean_deg - 1.0))))

    def compute_propagation_delay(
        self,
        admitted_mask: np.ndarray,
        is_malicious: np.ndarray,
        mempool_load_ratio: float,
        filter_latency_s: float = 0.0,
        block_size_mb: float = 1.25,
    ) -> float:
        """Compute expected block propagation delay B_prop (seconds) using BIP-152 compact relay.

        When the mempool is clean, nodes already hold almost all txs -> compact relay sends ~80 KB.
        When flooded by spam/DDoS, mempool evictions force fallback full-block / tx re-fetch
        transfers up to the full 1.25 MB block size, plus adversarial forwarding stalls.
        """
        active_idx = np.where(admitted_mask)[0]
        if len(active_idx) == 0:
            active_idx = np.arange(self.num_nodes)

        active_mal_ratio = float(is_malicious[active_idx].mean())
        active_bw = float(np.percentile(self.node_bw_mbps[active_idx], 35))
        mean_hop_s = float(self.base_hop_delay_s[active_idx].mean())

        # Compact block overhead scales with mempool churn/eviction under load
        effective_payload_mb = 0.08 + min(1.0, 0.45 * mempool_load_ratio) * (block_size_mb - 0.08)
        tx_delay_per_hop_s = (effective_payload_mb * 8.0) / max(8.0, active_bw)

        # Malicious relays deliberately delay / withhold announcements; congestion amplifies hop time
        stall_multiplier = 1.0 + 3.2 * active_mal_ratio + 2.0 * (active_mal_ratio ** 2)
        congestion_delay_s = 0.22 * (mempool_load_ratio ** 1.25)

        loss_multiplier = 1.0 / max(0.5, 1.0 - 5.0 * self.packet_loss * self.nominal_diameter_hops)

        hop_cost = (mean_hop_s + tx_delay_per_hop_s) * self.nominal_diameter_hops * stall_multiplier
        b_prop = (hop_cost + congestion_delay_s) * loss_multiplier + filter_latency_s

        noise = float(self.rng.normal(0.0, 0.025 * b_prop))
        return max(0.08, b_prop + noise)
