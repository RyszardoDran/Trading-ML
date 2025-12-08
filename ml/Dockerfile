# Multi-stage build for ML pipeline
# Stage 1: Base Python environment with dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies for ta-lib and other C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Stage 2: Install ML dependencies
FROM base as builder

WORKDIR /app

# Copy requirements file
COPY ml/requirements_ml.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements_ml.txt

# Stage 3: Runtime environment
FROM base as runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user site-packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH

# Copy ML project files
COPY ml/src ./src
COPY ml/tests ./tests
COPY ml/notebooks ./notebooks
COPY data ./data

# Create necessary directories
RUN mkdir -p logs models cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: run training pipeline
CMD ["python", "-m", "src.pipelines.training_pipeline"]
