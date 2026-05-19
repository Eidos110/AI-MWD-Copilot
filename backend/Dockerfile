FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/backend/

ENV PYTHONPATH=/app

WORKDIR /app

EXPOSE 8000

# NOTE: HEALTHCHECK removed — Railway's DOCKERFILE builder cannot parse the
# JSON-array form "["alt","http://localhost:8000/health",...]" used by older
# Docker-style HEALTHCHECK directives and returns "service unavailable" on every
# probe regardless of a correctly running container.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
