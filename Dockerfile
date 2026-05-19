FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["/app/start.sh"]

# Healthcheck targets the FastAPI /health endpoint.
# RAILWAY_PUBLIC_DOMAIN contains the current deployment URL (e.g. mwd-backend-production.up.railway.app).
# Falls back to localhost:8000 for local testing.
# Healthcheck: contact the FastAPI /health endpoint inside the running container.
# CMD-SHELL forces /bin/sh -c so the default call works without any env-var substitution.
HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=30s CMD-SHELL curl -sf http://localhost:8000/health || exit 1