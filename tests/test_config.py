from cpa_lppf.config import load_config


def test_default_config_loads():
    cfg = load_config(None)
    assert cfg.hfl.share_number >= cfg.hfl.reconstruction_threshold
    assert cfg.dataset.num_classes > 1
