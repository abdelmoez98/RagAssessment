FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install uv
RUN uv venv
RUN uv pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/store /app/policy_corpus /app/evaluation

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash"]
