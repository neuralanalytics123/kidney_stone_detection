FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project definition and install dependencies
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev

# Copy remaining application assets
COPY configs ./configs
COPY dashboard ./dashboard
COPY data ./data
COPY artifacts ./artifacts

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]