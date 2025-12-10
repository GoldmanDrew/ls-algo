LS-ALGO

LS-ALGO is an end-to-end Python pipeline for managing a rules-driven long/short trading strategy using a defined universe of tickers.
The system screens tickers, generates trading signals, maintains long/short positions, and monitors IBKR short-stock availability.
It is designed to run automatically once per day via GitHub Actions or locally via a helper script.

At a high level, the pipeline performs three core tasks:

1. ETF & Ticker Screening (etf_screener.py)

Screens a universe of tickers (from config/tickers.csv)

Calculates metrics (e.g., CAGRs, volatility, liquidity, weights) using historical datasets

Saves the daily screened output to data/etf_screened_today.csv

Produces upstream inputs for the trading algorithm

2. IBKR Trading Logic (ibkr_algo.py)

Reads the screened output

Applies the long/short allocation rules

Computes target positions and compares against prior state (data/positions_state.csv)

Determines required trades (buys, sells, covers, shorts)

Can be run in live mode or DRY_RUN mode

Intended to maintain the L/S book automatically through the IB API

3. Short-Stock Monitoring & Alerts (short_stock_monitor.py)

Downloads the public IBKR short-stock file (usa.txt)

Extracts borrow rates, rebates, and shares-available for relevant tickers

Compares to prior-day snapshots

Sends a daily email summarizing:

Borrow rate increases

Limited short supply

Threshold breaches (e.g., borrow caps, minimum shares available)

This gives you visibility into borrow constraints that may affect the execution or maintenance of short positions.

Repository Layout
.github/workflows/
    daily_pipeline.yml         # GitHub Actions workflow orchestrating the full daily run

config/
    etf_cagr.csv               # Supporting dataset (historical CAGRs or related metrics)
    tickers.csv                # Universe of tickers to screen and trade

data/
    etf_screened_today.csv     # Daily screener output
    positions_state.csv        # Last-known target positions for diffing

notebooks/
    leveredetfresearch.ipynb   # Research + exploratory analysis

scripts/
    run_daily_pipeline.sh      # Mirrors CI pipeline locally in identical order

etf_screener.py               # Step 1: screens ETF/ticker universe
ibkr_algo.py                  # Step 2: trading / long-short maintenance logic
short_stock_monitor.py        # Step 3: short borrow availability monitoring + alerts

requirements.txt              # Python dependencies
README.md                     # This file

Running Locally

Install dependencies:

pip install -r requirements.txt


Run any component individually:

python etf_screener.py
python ibkr_algo.py
python short_stock_monitor.py

Running the Entire Daily Pipeline Locally

The GitHub Actions workflow runs the pipeline in this exact order:

etf_screener.py

ibkr_algo.py

short_stock_monitor.py

To replicate the same behavior locally:

1. Export the required environment variables

IBKR FTP for short-stock file:
IBKR_FTP_HOST, IBKR_FTP_USER, IBKR_FTP_PASS, IBKR_FTP_FILE

Trade & monitoring params:
BORROW_CAP, MIN_SHARES_AVAILABLE, DRY_RUN

IB API connectivity:
IB_HOST, IB_PORT, IB_CLIENT_ID

Input/output paths:
SCREENED_CSV, SNAPSHOT_DIR

Email settings:
EMAIL_FROM, EMAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS

2. Run the pipeline wrapper:
./scripts/run_daily_pipeline.sh


This script:

Installs dependencies

Applies default environment variable values

Executes the three core scripts in the same sequence as CI

Allows you to validate the end-to-end behavior of the full system

Summary

LS-ALGO is evolving into a unified automation system that:

Screens the investable universe

Generates long/short portfolio targets

Executes or simulates trades

Monitors borrow constraints

Sends a consolidated daily email with trading and short-stock insights
