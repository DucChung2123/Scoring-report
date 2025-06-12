# Dockerfile chung cho ESG Scoring API và UI
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install dependencies
COPY requirements.txt .

# Install tất cả dependencies (API + UI)
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ application code
COPY src/ ./src/
COPY models/ ./models/
COPY settings/ ./settings/
COPY .env .env

# Expose ports (sẽ được override trong docker-compose)
EXPOSE 8000 8501

# Default command (sẽ được override trong docker-compose)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
