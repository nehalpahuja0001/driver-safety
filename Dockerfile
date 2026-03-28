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
    pyyaml

# Copy python module code
COPY *.py ./
COPY face_landmarker.task ./
COPY openenv.yaml ./

# Command to execute agent
CMD ["python", "inference.py"]
