from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

from cpa_lppf.utils.serialization import weighted_average_state_dicts


def fedavg(updates: list[tuple[Mapping[str, torch.Tensor], int | float]]) -> OrderedDict[str, torch.Tensor]:
    return weighted_average_state_dicts(updates)


def combine_updates(low: Mapping[str, torch.Tensor], high: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    out = OrderedDict()
    out.update(low)
    out.update(high)
    return out
