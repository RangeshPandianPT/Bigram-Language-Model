FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set PYTHONPATH to root directory
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000
EXPOSE 8501
