from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping

import torch

StateDict = OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]


def clone_state_dict(sd: Mapping[str, torch.Tensor], detach: bool = True, cpu: bool = True) -> OrderedDict[str, torch.Tensor]:
    out = OrderedDict()
    for k, v in sd.items():
        t = v.detach().clone() if detach else v.clone()
        if cpu:
            t = t.cpu()
        out[k] = t
    return out


def zeros_like_state_dict(sd: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, torch.zeros_like(v)) for k, v in sd.items())


def filter_state_dict(sd: Mapping[str, torch.Tensor], prefixes: Iterable[str]) -> OrderedDict[str, torch.Tensor]:
    prefixes = tuple(prefixes)
    return OrderedDict((k, v) for k, v in sd.items() if k.startswith(prefixes))


def state_dict_add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    keys = set(a) | set(b)
    out = OrderedDict()
    for k in sorted(keys):
        if k in a and k in b:
            out[k] = a[k] + b[k].to(a[k].device)
        elif k in a:
            out[k] = a[k].clone()
        else:
            out[k] = b[k].clone()
    return out


def state_dict_sub(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, a[k] - b[k].to(a[k].device)) for k in a.keys())


def state_dict_mul(a: Mapping[str, torch.Tensor], scalar: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, v * scalar) for k, v in a.items())


def weighted_average_state_dicts(items: list[tuple[Mapping[str, torch.Tensor], int | float]]) -> OrderedDict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot average an empty list")
    total = float(sum(w for _, w in items))
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for sd, w in items:
        scale = float(w) / max(total, 1e-12)
        for k, v in sd.items():
            if k not in out:
                out[k] = torch.zeros_like(v, dtype=v.dtype)
            out[k] += v * scale
    return out


def state_dict_to_vector(sd: Mapping[str, torch.Tensor], max_dim: int | None = None) -> torch.Tensor:
    vecs = [v.detach().float().reshape(-1).cpu() for v in sd.values()]
    if not vecs:
        return torch.empty(0)
    vec = torch.cat(vecs)
    if max_dim is not None and vec.numel() > max_dim:
        # deterministic uniform subsampling keeps feature dimensionality bounded
        idx = torch.linspace(0, vec.numel() - 1, max_dim).long()
        vec = vec[idx]
    return vec


def apply_delta_to_model(model: torch.nn.Module, delta: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    new_state = OrderedDict()
    for k, v in current.items():
        if k in delta:
            new_state[k] = v + delta[k].to(v.device)
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
