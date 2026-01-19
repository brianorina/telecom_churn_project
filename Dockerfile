# Base image with Java 11 already installed
FROM openjdk:11-jdk-slim

# Set working directory
WORKDIR /app

# Install Python 3 and pip (no Java install needed)
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8080

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
