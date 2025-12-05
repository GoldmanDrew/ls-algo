FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and config
COPY shortstockmonitor.py ./shortstockmonitor.py
COPY config ./config

CMD ["python", "shortstockmonitor.py"]
