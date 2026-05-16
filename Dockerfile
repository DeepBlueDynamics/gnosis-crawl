# Grub Crawler Dockerfile
# Use official Playwright Python image like grub

FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install additional dependencies (xvfb required for camoufox headless="virtual")
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iputils-ping \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain + C linker for grub_md native extension
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (chromium) to match the installed Python package version
RUN playwright install --with-deps chromium

# Install Patchright browser (stealth Chromium fork)
RUN patchright install chromium || true

# Fetch Camoufox browser binary once — directly into the runtime user's
# cache dir. Placed BEFORE COPY app/ / site/ / VERSION so app-code changes
# don't invalidate this 700MB layer.
# Needs root for GeoLite2 MMDB install into system packages; HOME redirect
# puts the browser zip into /home/app/.cache/camoufox/ where USER app can
# read it at runtime.
RUN HOME=/home/app python -m camoufox fetch \
    && chown -R app:app /home/app/.cache

# Build and install grub_md Rust native extension
COPY grub_md/ ./grub_md/
RUN cd grub_md && maturin build --release \
    && pip install --no-cache-dir target/wheels/*.whl \
    && rm -rf target

# Copy application code
COPY app/ ./app/

# Copy embedded landing page (grub-site)
COPY site/ ./site/

# Copy VERSION file — single source of truth for /health version
COPY VERSION ./VERSION

# Create storage directory
RUN mkdir -p storage && chown -R app:app storage

# Switch to non-root user for runtime
USER app

# Expose port
EXPOSE 6792

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-6792}/health || exit 1

# Run application — respect PORT env var (Cloud Run sets PORT=8080)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-6792}
