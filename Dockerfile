FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VALSTORM_DEFAULT_SANDBOX=cloud \
    PORT=8660

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy monorepo dependencies referenced by [tool.uv.sources]
COPY packages/valstorm-sanitizer /workspace/packages/valstorm-sanitizer
COPY packages/client-python /workspace/packages/client-python

# Copy dependency specification and install Python packages
COPY apps/agent-runtime/pyproject.toml /workspace/apps/agent-runtime/
WORKDIR /workspace/apps/agent-runtime
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application source code
COPY apps/agent-runtime /workspace/apps/agent-runtime

# Expose cloud runtime port
EXPOSE 8660

# Health check endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8660/health || exit 1

# Start Cloud Agent Runtime server in Cloud MicroVM mode
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8660", "--mode", "cloud", "--default-sandbox", "cloud"]
