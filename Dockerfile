FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VALSTORM_DEFAULT_SANDBOX=cloud \
    PORT=8660

WORKDIR /workspace

# Install system dependencies and create unprivileged runtime user
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 valstorm \
    && useradd -u 1000 -g valstorm -m -s /bin/bash valstorm \
    && mkdir -p /workspace && chown valstorm:valstorm /workspace

# Install uv package manager for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy monorepo dependencies referenced by [tool.uv.sources]
COPY --chown=valstorm:valstorm packages/valstorm-sanitizer /workspace/packages/valstorm-sanitizer
COPY --chown=valstorm:valstorm packages/client-python /workspace/packages/client-python

# Copy dependency specification and install locked Python packages
COPY --chown=valstorm:valstorm apps/agent-runtime/pyproject.toml apps/agent-runtime/uv.lock* /workspace/apps/agent-runtime/
WORKDIR /workspace/apps/agent-runtime
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY --chown=valstorm:valstorm apps/agent-runtime /workspace/apps/agent-runtime
RUN ln -s /workspace/apps/agent-runtime /app

USER valstorm

# Expose cloud runtime port
EXPOSE 8660

# Health check endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8660/health || exit 1

# Start Cloud Agent Runtime server in Cloud MicroVM mode
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8660", "--mode", "cloud", "--default-sandbox", "cloud"]
