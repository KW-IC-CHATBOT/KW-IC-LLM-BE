# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    netcat-traditional \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy .env file
COPY .env .

# Install sentence transformers for embeddings
RUN pip install -r requirements.txt --no-cache-dir

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 35509

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "35504", "--workers", "1"]