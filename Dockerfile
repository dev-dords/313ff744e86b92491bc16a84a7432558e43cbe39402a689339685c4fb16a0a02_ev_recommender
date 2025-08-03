FROM apache/airflow:3.0.3

USER airflow

# Optional: Install uv as airflow user (if not already available)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy pyproject & lock files
COPY --chown=airflow:0 pyproject.toml uv.lock* ./

# Compile & install dependencies
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

# Copy project source and data
COPY --chown=airflow:0 src/ ./src/
COPY --chown=airflow:0 data/bronze/ ./data/bronze/
