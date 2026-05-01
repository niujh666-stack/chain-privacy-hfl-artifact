from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

LOW_PREFIXES = ("trunk",)
HIGH_PREFIXES = ("head",)


def is_low_key(key: str) -> bool:
    return key.startswith(LOW_PREFIXES)


def is_high_key(key: str) -> bool:
    return key.startswith(HIGH_PREFIXES)


def split_state_dict(sd: Mapping[str, torch.Tensor]) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    low = OrderedDict((k, v) for k, v in sd.items() if is_low_key(k))
    high = OrderedDict((k, v) for k, v in sd.items() if is_high_key(k))
    return low, high


def merge_low_high(
    base: Mapping[str, torch.Tensor], low: Mapping[str, torch.Tensor], high: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    out = OrderedDict()
    for k, v in base.items():
        if k in low:
            out[k] = low[k]
        elif k in high:
            out[k] = high[k]
        else:
            out[k] = v
    return out


def delta_split(local: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]):
    delta = OrderedDict((k, local[k].detach().cpu() - global_state[k].detach().cpu()) for k in local.keys())
    return split_state_dict(delta)
