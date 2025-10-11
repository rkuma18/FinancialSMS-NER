FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy everything from the frontend directory
COPY frontend/ .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Run the streamlit app (adjust the filename if needed)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]