# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (libgomp1 is needed for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Install CPU-only PyTorch first to save build time and space
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
