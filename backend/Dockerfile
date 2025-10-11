# backend/Dockerfile

# Base image
FROM python:3.11-slim

# Install Git and Git LFS (Requires apt to install OS-level dependencies)
# Yeh steps ab Pre-Deploy Command ka kaam karenge
RUN apt-get update && apt-get install -y git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for non-interactive mode
ENV PYTHONUNBUFFERED 1
ENV POETRY_NO_INTERACTION 1

# Set the working directory inside the container
WORKDIR /app

# Crucial Step: Pull LFS files inside the container context
# Isse artifacts files sync ho jayengi aur 'pyproject.toml' error theek ho jayega
RUN git lfs pull --include "*"

# 1. Install Poetry
RUN pip install poetry

# 2. Copy Poetry dependency files 
COPY backend/pyproject.toml backend/poetry.lock /app/

# 3. Install dependencies from lock file
RUN poetry install --no-root --only main

# 4. Copy the large model artifacts from the repository root
COPY artifacts /app/artifacts

# 5. Copy the main backend application code
COPY backend /app/backend

# 6. Expose the port (FastAPI's default)
EXPOSE 8000

# 7. Start the Uvicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]