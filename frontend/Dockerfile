# Dockerfile for Streamlit Client

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create and set the working directory
WORKDIR $APP_HOME

# Copy and install dependencies
# Streamlit dependencies only (e.g., streamlit, requests)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code and the rest of the files (e.g., requirements.txt)
# We copy the entire repo, but only the Streamlit files are needed
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# The command to run the Streamlit application
# Ensure your Streamlit file is named 'streamlit_app.py' and is in the root directory.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]