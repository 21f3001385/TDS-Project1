FROM python:3.11-slim

WORKDIR /app

# Install Node.js and npm for Prettier
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /data

EXPOSE 8000

CMD ["python", "main.py"]