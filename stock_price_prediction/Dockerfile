# stock_price_prediction/Dockerfile
FROM python:3.9

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y libopenblas-dev cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "main.py"]
