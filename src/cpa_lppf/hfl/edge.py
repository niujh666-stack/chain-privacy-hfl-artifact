from __future__ import annotations

from collections import defaultdict


class EdgeNode:
    def __init__(self, edge_id: int, malicious: bool = False):
        self.edge_id = int(edge_id)
        self.malicious = bool(malicious)
        self.received = defaultdict(dict)

    def receive_share(self, round_idx: int, client_id: int, share):
        self.received[int(round_idx)][int(client_id)] = share

    def leak_share(self, round_idx: int, client_id: int):
        if not self.malicious:
            return None
        return self.received.get(int(round_idx), {}).get(int(client_id))
