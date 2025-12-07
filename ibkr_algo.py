# pip install ib_insync
import pandas as pd
from ib_insync import IB, Stock
import time

def connect_ibkr(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1
) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def fetch_short_data_for_etf(ib: IB, symbol: str, exchange: str = "SMART") -> dict:
    """
    Use generic tick 236 (shortable) to get fee rate and shares.

    Note: This is indicative only and can change intraday.
    """
    contract = Stock(symbol, exchange, "USD")
    ticker = ib.reqMktData(contract, genericTickList="236", snapshot=True)
    # Wait a bit for data to arrive
    ib.sleep(1.5)

    # ib_insync exposes shortableShares as a list of tiers, each with:
    #   .shares, .feeRate, .tier
    short_list = getattr(ticker, "shortableShares", None)

    if not short_list:
        return {
            "ETF": symbol,
            "borrow_current": float("nan"),
            "shares_available": 0,
        }

    # Take the first tier as indicative
    tier0 = short_list[0]
    return {
        "ETF": symbol,
        "borrow_current": tier0.feeRate / 100.0,  # convert % -> decimal
        "shares_available": tier0.shares,
    }


def get_ibkr_borrow_snapshot(etf_list: Iterable[str]) -> pd.DataFrame:
    ib = connect_ibkr()
    try:
        records = []
        for sym in etf_list:
            rec = fetch_short_data_for_etf(ib, sym)
            records.append(rec)
            # be nice to the API throttle
            time.sleep(0.2)
    finally:
        ib.disconnect()

    df = pd.DataFrame(records)

    # For now no trend filter -> set borrow_spiking=False
    df["borrow_spiking"] = False
    return df

universe_df = pd.read_csv("etf_universe.csv")
print(universe_df.head())

borrow_df = get_ibkr_borrow_snapshot(universe_df["ETF"].unique())
universe_metrics = (
    universe_df
    .merge(cagr_df, on="ETF", how="left")
    .merge(borrow_df, on="ETF", how="left")
)
screened = screen_universe_for_algo(universe_metrics)
tradable_today = screened[screened["include_for_algo"]]
