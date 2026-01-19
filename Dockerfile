# ---------- Base image with Java 11 ----------
FROM eclipse-temurin:11-jdk

# Set JAVA_HOME so PySpark can find Java
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Install Python3, venv, pip, and build tools
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY . /app

# Install Python dependencies inside virtual environment
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
