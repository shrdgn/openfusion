FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY openfusion ./openfusion

RUN pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin --uid 1000 openfusion \
    && chown -R openfusion:openfusion /app

USER openfusion

# Boots zero-config (Budget preset) and serves the playground; pass
# OPENROUTER_API_KEY, mount an openfusion.yaml at /app, or set the key in the UI.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('OPENFUSION_PORT', '8000') + '/healthz', timeout=2)"

CMD ["openfusion", "web", "--host", "0.0.0.0", "--port", "8000"]
