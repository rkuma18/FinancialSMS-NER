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

# 2. Copy Poetry dependency files 
# Since Docker Context is repo root, we use backend/pyproject.toml
COPY backend/pyproject.toml backend/poetry.lock /app/

# 3. Install dependencies from lock file
RUN poetry install --no-root --only main

# 4. Copy the large model artifacts from the repository root
# 'artifacts' folder repo root par hai, aur woh direct accessible hai
COPY artifacts /app/artifacts

# 5. Copy the main backend application code
# Backend folder ko /app/backend ke andar copy kar rahe hain
COPY backend /app/backend

# 6. Expose the port (FastAPI's default)
EXPOSE 8000

# 7. Start the Uvicorn server
# Application ab /app/backend/main.py se chalegi
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]