FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Clone the repository WITH LFS files during build
# This bypasses the issue of Render not pulling LFS files
RUN git clone https://github.com/rkuma18/financial-sms-ner-app.git /tmp/repo && \
    cd /tmp/repo && \
    git lfs pull && \
    cp -r /tmp/repo/* /app/ && \
    cp -r /tmp/repo/.* /app/ 2>/dev/null || true && \
    rm -rf /tmp/repo

# Install Poetry
RUN pip install poetry

# Install Python dependencies
RUN cd backend && poetry install --no-root --only main

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]