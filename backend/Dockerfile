FROM python:3.11-slim

RUN apt-get update && apt-get install -y git git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

ENV PYTHONUNBUFFERED 1
ENV POETRY_NO_INTERACTION 1

WORKDIR /app

# Copy only what's needed for LFS pull
COPY .git /app/.git
COPY .gitattributes /app/.gitattributes

# Pull LFS files
RUN git lfs pull

# Install Poetry
RUN pip install poetry

# Copy dependency files (which should now be available)
COPY backend/pyproject.toml backend/poetry.lock /app/backend/

# Install dependencies (cached unless these files change)
RUN cd backend && poetry install --no-root --only main

# Copy the rest of the application
COPY artifacts /app/artifacts
COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]