# Dockerfile

FROM python:3.11-slim

# Install system dependencies (Git and Git LFS)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Copy the entire repository content into the container
COPY . /app

# Ensure git trusts the copied working tree, then fetch LFS assets.
RUN git config --global --add safe.directory /app && \
    git lfs pull --include "*"

# Install Poetry
RUN pip install poetry

# Install Python dependencies from the backend module
RUN cd backend && poetry install --no-root --only main

EXPOSE 8000

# Start the Uvicorn server (main:app is inside the backend directory)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
