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

# Healthcheck: /health must return HTTP 200 inside the running container.
# Use "local" shell form so the command string is executed via /bin/sh -c.
HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=30s CMD /bin/sh -c "curl -sf --max-time 3 http://localhost:8000/health && exit 0 || exit 1"