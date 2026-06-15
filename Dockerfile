FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENFUSION_CONFIG=/app/openfusion.yaml

COPY pyproject.toml README.md LICENSE ./
COPY openfusion ./openfusion
COPY openfusion.yaml.example ./openfusion.yaml.example

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["openfusion", "--host", "0.0.0.0", "--port", "8000"]
