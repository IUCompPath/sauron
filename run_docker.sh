#!/bin/bash
# Helper script to run Docker container with E: drive mounted
# Usage: ./run_docker.sh [command]

set -e

# Detect if running in WSL or native Windows
if [[ -d "/mnt/e" ]]; then
    # WSL environment
    E_DRIVE_PATH="/mnt/e"
    echo "Detected WSL environment, using path: $E_DRIVE_PATH"
elif [[ -d "/e" ]]; then
    # Alternative WSL path
    E_DRIVE_PATH="/e"
    echo "Detected WSL environment (alternative), using path: $E_DRIVE_PATH"
else
    # Native Windows Docker (use Windows path format)
    E_DRIVE_PATH="E:/"
    echo "Detected native Windows Docker, using path: $E_DRIVE_PATH"
fi

# Build the image if it doesn't exist
if ! docker images | grep -q "aegis"; then
    echo "Building Docker image..."
    docker build -t aegis .
fi

# Default command (interactive bash shell)
CMD="${1:-/bin/bash}"

# Run the container with E: drive mounted
echo "Running Docker container with E: drive mounted at /data"
echo "Command: $CMD"
docker run -it --rm \
    --gpus all \
    --shm-size 512g \
    --memory 96g \
    --cpus 32 \
    -v "$E_DRIVE_PATH:/data:rw" \
    -v "$(pwd)/results:/app/results:rw" \
    -v "$(pwd)/Data:/app/Data:rw" \
    -w /app \
    aegis \
    $CMD

