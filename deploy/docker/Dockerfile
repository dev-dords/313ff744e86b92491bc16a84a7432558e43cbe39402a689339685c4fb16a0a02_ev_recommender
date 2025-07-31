
FROM python:3.12.11

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Create a virtual environment inside the image
RUN uv venv /opt/venv
RUN mkdir data
RUN mkdir data/bronze
RUN mkdir models
# Activate the venv for all future RUN/CMD instructions
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy pyproject and lock file
COPY pyproject.toml uv.lock* ./

# Install dependencies in the venv (no --system needed)
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/bronze/ ./data/bronze/
# Set default command
CMD ["python", "src/run_pipeline.py"]
