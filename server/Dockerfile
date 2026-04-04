FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required by mediapipe and opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    mediapipe \
    numpy \
    scipy \
    pydantic \
    openai \
    pyyaml \
    fastapi \
    uvicorn \
    websockets

# Copy everything from repository root
COPY . ./

# Expose port
EXPOSE 7860

# Command to execute FastAPI server containing the WebSocket endpoints
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
