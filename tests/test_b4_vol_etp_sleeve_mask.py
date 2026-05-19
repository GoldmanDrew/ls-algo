"""Vol-B4 sleeve mask: volatility ETPs need not pass ``inverse_shortable``."""

import pandas as pd

from generate_trade_plan import _in_b4_volatility_etp_sleeve_mask


def test_vol_etp_b4_slice_true_when_inverse_shortable_false():
    eligible = pd.DataFrame(
        {
            "ETF": ["UVIX"],
            "Underlying": ["SVIX"],
            "Delta": [-2.0],
            "Delta_product_class": ["volatility_etp"],
            "inverse_shortable": [False],
        }
    )
    idx = eligible.index
    all_true = pd.Series(True, index=idx)
    mask = _in_b4_volatility_etp_sleeve_mask(
        eligible,
        b4_borrow_ok=all_true,
        b4_edge_ok=all_true,
        b4_vol_ok=all_true,
        b4_not_excluded=all_true,
        in_flow_program=pd.Series(False, index=idx),
        in_b4_core=pd.Series(False, index=idx),
    )
    assert bool(mask.iloc[0])


def test_vol_etp_b4_slice_false_when_already_in_b4_core():
    eligible = pd.DataFrame(
        {
            "ETF": ["UVIX"],
            "Underlying": ["SVIX"],
            "Delta": [-2.0],
            "Delta_product_class": ["volatility_etp"],
            "inverse_shortable": [True],
        }
    )
    idx = eligible.index
    all_true = pd.Series(True, index=idx)
    mask = _in_b4_volatility_etp_sleeve_mask(
        eligible,
        b4_borrow_ok=all_true,
        b4_edge_ok=all_true,
        b4_vol_ok=all_true,
        b4_not_excluded=all_true,
        in_flow_program=pd.Series(False, index=idx),
        in_b4_core=pd.Series(True, index=idx),
    )
    assert not bool(mask.iloc[0])
