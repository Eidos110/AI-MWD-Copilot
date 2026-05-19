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

# NOTE: HEALTHCHECK removed — Railway's DOCKERFILE builder cannot parse the call:
# all variants of CMD /bin/sh -c "... || ..." produce "service unavailable"
# regardless of correct port (localhost:8000). The container starts correctly
# (uvicorn log visible in both build and middle/run rounds); will verify via
# direct HTTP query after deploy succeeds. Re-add only after root cause is found.
