"""Histogram JSON for bootstrap net edge."""
import json

import numpy as np

from screener_v2_fields import _net_edge_hist_json


def test_net_edge_hist_json_roundtrip():
    rng = np.random.default_rng(3)
    x = rng.normal(0.12, 0.04, size=500)
    s = _net_edge_hist_json(x, n_bins=12)
    assert s is not None
    d = json.loads(s)
    assert "e" in d and "c" in d
    assert len(d["e"]) == len(d["c"]) + 1
    assert sum(d["c"]) >= 400  # most mass inside 1–99% range used for bins
