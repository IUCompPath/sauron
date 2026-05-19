# Use an official NVIDIA CUDA runtime as a parent image
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenslide-dev \
    sed \
    grep \
    && rm -rf /var/lib/apt/lists/*

# Fix for "externally managed" environment error:
# Remove the marker file to allow system-wide package installation
# This is necessary because the base image enforces PEP 668, but we want to install into the system environment in Docker.
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED



# Install uv
RUN pip3 install --no-cache-dir uv

# Copy the project files into the container
COPY . /app

# Create a filtered requirements file excluding torch/torchvision/torchaudio
# We exclude these to avoid overwriting the optimized versions provided by the NVIDIA base image.
# We use python to carefully filter only the core torch packages, preserving torch_geometric etc.
RUN python3 -c "lines = [l.strip() for l in open('/app/requirements.txt') if l.strip()]; \
    filtered = [l for l in lines if not (l == 'torch' or l.startswith(('torch==', 'torch>=', 'torch<', 'torch>', 'torch~=', 'torchvision', 'torchaudio')))]; \
    print('\n'.join(filtered))" > /tmp/requirements_no_torch.txt

# Install Python dependencies using uv (system-wide)
RUN uv pip install --system --no-cache -r /tmp/requirements_no_torch.txt

# Install the package using uv
RUN uv pip install --system --no-cache /app

# Set the entrypoint
# Using /bin/bash allows interactive use.
ENTRYPOINT ["/bin/bash"]

# To mount your E: drive when running the container, use:
# 
# For WSL/Linux:
#   docker run -it --gpus all -v /mnt/e:/data /app /bin/bash
#
# For Windows (native Docker Desktop):
#   docker run -it --gpus all -v E:/:/data /app /bin/bash
#
# Or use docker-compose.yml (see docker-compose.yml file)

#
# Or use docker-compose.yml (see docker-compose.yml file)
