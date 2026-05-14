"""B4 no-IBKR proxy rows include volatility ETPs without requiring inverse_shortable."""

import pandas as pd

from plot_proposed_trades import append_no_ibkr_share_screened_rows


def test_b4_proxy_includes_vol_etp_when_inverse_shortable_false():
    screened = pd.DataFrame(
        [
            {
                "ETF": "UVIX",
                "Underlying": "VIX",
                "Beta": -1.98,
                "is_yieldboost": False,
                "inverse_shortable": False,
                "Beta_product_class": "volatility_etp",
                "shares_available": 0.0,
                "exclude_no_shares": True,
                "net_decay_annual": 0.1,
            }
        ]
    )
    out = append_no_ibkr_share_screened_rows(
        pd.DataFrame(),
        screened,
        bucket="b4",
    )
    assert not out.empty
    assert (out["ETF"] == "UVIX").any()
    assert bool(out.loc[out["ETF"] == "UVIX", "_no_ibkr_shares_proxy"].iloc[0])
    assert out.loc[out["ETF"] == "UVIX", "sleeve"].iloc[0] == "inverse_decay_bucket4"
