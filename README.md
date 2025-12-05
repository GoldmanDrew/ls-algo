# Short Stock Monitor

Small Python service that:
- Downloads IBKR public shortstock file (`usa.txt`) once a day
- Extracts borrow rate, rebate, and shares available for a set of tickers
- Compares to previous day and emails alerts when thresholds are breached

## Layout

- `shortstockmonitor.py` – main script
- `config/ym_tickers.csv` – list of tickers to watch
- `requirements.txt` – Python dependencies
- `Dockerfile` – container build
- `k8s/cronjob.yaml` – example Kubernetes CronJob

## Running locally

```bash
pip install -r requirements.txt
python shortstockmonitor.py
