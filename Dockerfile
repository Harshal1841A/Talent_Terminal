# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Copy the current directory contents into the container
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose port 7860 which HF Spaces expects
EXPOSE 7860

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
