from collections import OrderedDict

import torch

from cpa_lppf.defenses.lgs import LowSensitiveGradientSanitizer, state_norm


def test_lgs_clipping_no_noise():
    update = OrderedDict({"trunk.w": torch.ones(10)})
    lgs = LowSensitiveGradientSanitizer(clip_norm=1.0, noise_multiplier=0.0)
    out = lgs(update)
    assert float(state_norm(out)) <= 1.0001
