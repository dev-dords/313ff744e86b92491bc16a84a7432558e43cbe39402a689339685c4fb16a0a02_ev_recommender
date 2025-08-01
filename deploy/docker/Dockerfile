
FROM python:3.12.11

# Set working directory
WORKDIR /app

RUN pip install --no-cache-dir uv

RUN uv venv /opt/venv

RUN mkdir data
RUN mkdir data/bronze
RUN mkdir models

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml uv.lock* ./

RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

COPY src/ ./src/
COPY data/bronze/ ./data/bronze/

# Set default command
# CMD ["python", "src/run_pipeline.py"]

CMD ["tail", "-f", "/dev/null"]
