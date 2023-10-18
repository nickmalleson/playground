FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3-tk \
    build-essential \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

