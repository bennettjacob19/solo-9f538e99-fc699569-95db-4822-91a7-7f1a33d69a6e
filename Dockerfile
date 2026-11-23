FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    STRIPES=16 \
    OFFLINE_DISK=none \
    FSYNC=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The named volume must be writable by the unprivileged runtime user. Create the
# user and chown the data dir *before* VOLUME so the ownership is baked in.
RUN mkdir -p /data \
    && useradd --system --uid 10001 --home-dir /srv array \
    && chown -R array:array /srv /data

VOLUME ["/data"]
USER array

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
