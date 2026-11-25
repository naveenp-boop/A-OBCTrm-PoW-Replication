#!/usr/bin/env bash
set -euo pipefail

# Run inside privileged Linux environment with Mininet + OVS installed
sudo service openvswitch-switch start || true
sudo mn -c || true
sudo python3 mininet_testbed/topology_wan.py
