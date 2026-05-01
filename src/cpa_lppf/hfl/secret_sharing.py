from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

PRIME = 2_147_483_647  # Mersenne prime, safe for int64 arithmetic in this artifact.


def _encode_tensor(t: torch.Tensor, scale: float, prime: int = PRIME) -> torch.Tensor:
    q = torch.round(t.detach().cpu().float() * scale).to(torch.int64)
    return torch.remainder(q, prime)


def _decode_tensor(q: torch.Tensor, scale: float, prime: int = PRIME) -> torch.Tensor:
    q = q.to(torch.int64)
    signed = torch.where(q > prime // 2, q - prime, q)
    return signed.float() / scale


def _mod_inv(x: int, prime: int = PRIME) -> int:
    return pow(int(x) % prime, -1, prime)


class ShamirTensorSecretSharing:
    """Threshold secret sharing for tensor state dictionaries.

    Floating-point tensors are quantized into a finite field.  This is sufficient
    for simulation and reviewer inspection; production cryptography requires a
    dedicated audited implementation.
    """

    def __init__(self, num_shares: int = 3, threshold: int = 2, scale: float = 1_000_000.0, prime: int = PRIME, seed: int = 0):
        if threshold < 2:
            raise ValueError("threshold must be at least 2")
        if num_shares < threshold:
            raise ValueError("num_shares must be >= threshold")
        self.num_shares = int(num_shares)
        self.threshold = int(threshold)
        self.scale = float(scale)
        self.prime = int(prime)
        self.seed = int(seed)

    def share_tensor(self, tensor: torch.Tensor, name: str = "") -> list[tuple[int, torch.Tensor]]:
        secret = _encode_tensor(tensor, self.scale, self.prime)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + abs(hash(name)) % 1_000_000)
        coeffs = [secret]
        for _ in range(self.threshold - 1):
            coeffs.append(torch.randint(0, self.prime, size=secret.shape, generator=gen, dtype=torch.int64))
        shares = []
        for x in range(1, self.num_shares + 1):
            y = torch.zeros_like(secret, dtype=torch.int64)
            power = 1
            for coeff in coeffs:
                y = torch.remainder(y + coeff * power, self.prime)
                power = (power * x) % self.prime
            shares.append((x, y))
        return shares

    def reconstruct_tensor(self, shares: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        if len(shares) < self.threshold:
            raise ValueError(f"need at least {self.threshold} shares, got {len(shares)}")
        selected = shares[: self.threshold]
        acc = torch.zeros_like(selected[0][1], dtype=torch.int64)
        for i, (x_i, y_i) in enumerate(selected):
            num = 1
            den = 1
            for j, (x_j, _) in enumerate(selected):
                if i == j:
                    continue
                num = (num * (-x_j)) % self.prime
                den = (den * (x_i - x_j)) % self.prime
            coeff = (num * _mod_inv(den, self.prime)) % self.prime
            acc = torch.remainder(acc + y_i * coeff, self.prime)
        return _decode_tensor(acc, self.scale, self.prime)

    def share_state_dict(self, state: Mapping[str, torch.Tensor]) -> list[tuple[int, OrderedDict[str, torch.Tensor]]]:
        per_key: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for name, tensor in state.items():
            per_key[name] = self.share_tensor(tensor, name=name)
        out = []
        for idx in range(self.num_shares):
            x = idx + 1
            share = OrderedDict((name, shares[idx][1]) for name, shares in per_key.items())
            out.append((x, share))
        return out

    def reconstruct_state_dict(self, shares: list[tuple[int, Mapping[str, torch.Tensor]]]) -> OrderedDict[str, torch.Tensor]:
        if len(shares) < self.threshold:
            raise ValueError(f"need at least {self.threshold} shares, got {len(shares)}")
        keys = list(shares[0][1].keys())
        out = OrderedDict()
        for key in keys:
            tensor_shares = [(x, share[key]) for x, share in shares]
            out[key] = self.reconstruct_tensor(tensor_shares)
        return out
