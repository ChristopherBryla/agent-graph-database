# ---------------------------------------------------------------------------
# Genesys API — FastAPI service
# ---------------------------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/
COPY data/ data/

# Config lives under src/genesys/component_manager/
ENV PYTHONPATH=/app/src

# Create the data directory so SQLite has a place to write
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "genesys.app:app", "--host", "0.0.0.0", "--port", "8000"]
