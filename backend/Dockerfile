FROM python:3.11-slim

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy everything
COPY . /app/

# Install dependencies using pip
RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]