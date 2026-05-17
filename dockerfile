FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/output && chmod 777 /app/logs /app/output

CMD ["python", "mosquito_trajectory_prediction.py"]
