# Base image for Python applications (A lightweight image is preferred)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create and set the working directory
WORKDIR $APP_HOME

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# --host 0.0.0.0 is crucial for Docker containers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]