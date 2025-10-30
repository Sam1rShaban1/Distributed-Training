# Use a PyTorch base image with CUDA support (adjust CUDA version if needed)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install any additional dependencies (e.g., MPI, NCCL, etc.)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio

# Copy the project files into the container
COPY . /app

# Expose port for communication
EXPOSE 29500

# Default command to run the entrypoint script
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
