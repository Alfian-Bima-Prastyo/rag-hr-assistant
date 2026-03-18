FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app/
COPY documents/ ./documents/

# Expose FastAPI port
EXPOSE 8000

# Jalankan ingestion dulu, lalu start FastAPI
CMD ["sh", "-c", "python -m app.ingestion && uvicorn app.main:app --host 0.0.0.0 --port 8000"]