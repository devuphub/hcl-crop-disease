# Production Dockerfile for Crop Disease Detection Platform
FROM python:3.11-slim

# Avoid writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU directly first for fast build caching
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Default command launches the Streamlit web dashboard
CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
