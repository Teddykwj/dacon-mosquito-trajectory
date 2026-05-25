FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p /app/logs /app/output && chmod 777 /app/logs /app/output

CMD ["python", "mosquito_trajectory_prediction.py"]
