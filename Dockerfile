FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes reuse this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Train once at build time; every container then serves the same model.
COPY train.py app.py ./
RUN python train.py model

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 5050
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5050/health', timeout=2)"

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "--worker-tmp-dir", "/dev/shm", "--access-logfile", "-", "app:app"]
