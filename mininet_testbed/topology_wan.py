"""50-Node Emulated WAN Mininet Topology with OVS Switches and tc netem.

Note: Requires a Linux host with Mininet (`mn`), Open vSwitch (`ovs-vswitchd`),
and `iproute2` (`tc netem`). Parameters match Table 4 and Section 5.1:
- 50 containerized / isolated light-PoW miner nodes
- Heavy-tailed WAN delay via tc netem log-normal emulation (median ~90ms; mu_ln=4.5, sigma_ln=0.9)
- 0.5% Bernoulli packet loss and asymmetric edge/backbone bandwidth caps
"""

import math
import random
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink
from mininet.log import setLogLevel, info


class WANBlockchainTopo(Topo):
    """50-node regional P2P WAN topology over Open vSwitch core switches."""

    def build(self, n_nodes=50, seed=42):
        rng = random.Random(seed)
        switches = [self.addSwitch(f"s{r+1}") for r in range(5)]

        # Core ring + chord links across 5 WAN regions
        for i in range(5):
            j = (i + 1) % 5
            self.addLink(
                switches[i],
                switches[j],
                cls=TCLink,
                bw=100,
                delay="45ms",
                jitter="12ms",
                loss=0.2,
            )

        # Attach 50 miner hosts with log-normal WAN delays (mu_ln=4.5, sigma_ln=0.9)
        for idx in range(n_nodes):
            sw = switches[idx % 5]
            host = self.addHost(f"m{idx+1}")
            delay_ms = max(15.0, min(600.0, math.exp(rng.gauss(4.5, 0.9)) / 2.0))
            bw = 50 if idx < 8 else rng.randint(10, 20)
            self.addLink(
                host,
                sw,
                cls=TCLink,
                bw=bw,
                delay=f"{delay_ms:.1f}ms",
                jitter=f"{0.2 * delay_ms:.1f}ms",
                loss=0.5,
            )


def run():
    setLogLevel("info")
    topo = WANBlockchainTopo(n_nodes=50, seed=42)
    net = Mininet(topo=topo, controller=OVSController, link=TCLink)
    net.start()
    info("*** 50-node WAN blockchain testbed initialized\n")
    net.pingAll(timeout="1")
    net.stop()


if __name__ == "__main__":
    run()
