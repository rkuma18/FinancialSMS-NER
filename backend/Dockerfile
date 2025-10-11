# backend/Dockerfile

# Base image
FROM python:3.11-slim

# Set environment variables for non-interactive mode
ENV PYTHONUNBUFFERED 1
ENV POETRY_NO_INTERACTION 1

# Set the working directory inside the container
WORKDIR /app

# 1. Install Poetry
RUN pip install poetry

# 2. Copy Poetry dependency files (pyproject.toml and poetry.lock)
# Render's Root Directory is 'backend', so paths are relative to 'backend'
COPY pyproject.toml poetry.lock /app/

# 3. Install dependencies from lock file
# "--no-root" ensures it only installs dependencies, not the current package
RUN poetry install --no-root --only main

# 4. Copy the large model artifacts from the repository root (../artifacts)
# This is the crucial fix for the "Model directory not found" error
COPY ../artifacts /app/artifacts

# 5. Copy the main application code (main.py and other source files)
COPY . /app

# 6. Expose the port (FastAPI's default)
EXPOSE 8000

# 7. Start the Uvicorn server
# The --reload flag should be removed for production/deployment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]