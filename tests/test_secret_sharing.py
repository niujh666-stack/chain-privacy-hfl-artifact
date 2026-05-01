from collections import OrderedDict

import torch

from cpa_lppf.hfl.secret_sharing import ShamirTensorSecretSharing


def test_shamir_state_dict_roundtrip():
    ss = ShamirTensorSecretSharing(num_shares=3, threshold=2, seed=1)
    state = OrderedDict({"head.0.weight": torch.randn(4, 3) * 0.01, "head.0.bias": torch.randn(4) * 0.01})
    shares = ss.share_state_dict(state)
    recon = ss.reconstruct_state_dict([shares[0], shares[2]])
    for key in state:
        assert torch.allclose(state[key], recon[key], atol=2e-6)
