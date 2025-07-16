# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY packages.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Build the React Native web app
RUN cd FaceAgeApp && \
    npm install && \
    npx expo export --platform web

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"] 