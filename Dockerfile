FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN uv sync

COPY . .

RUN mkdir -p /app/store /app/policy_corpus /app/evaluation

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash"]
