# Docker Usage with E: Drive

This guide explains how to run the AEGIS Docker container with your E: drive mounted.

**Note:** The Dockerfile uses `uv` for fast Python package installation. PyTorch dependencies are excluded from requirements.txt as they're already included in the base NVIDIA PyTorch image.

## Quick Start

### Option 1: Using the Helper Script (Recommended)

**For WSL/Linux:**
```bash
./run_docker.sh
```

**For Windows (Command Prompt/PowerShell):**
```cmd
run_docker.bat
```

### Option 2: Using Docker Compose (Recommended for Development)

**Build and run:**
```bash
docker-compose up -d
docker-compose exec aegis /bin/bash
```

**Or build and run in one command:**
```bash
docker-compose up --build
```

**To stop:**
```bash
docker-compose down
```

**To rebuild after changes:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Option 3: Manual Docker Run

**For WSL:**
```bash
docker build -t aegis .
docker run -it --gpus all -v /mnt/e:/data aegis /bin/bash
```

**For Windows (native Docker Desktop):**
```bash
docker build -t aegis .
docker run -it --gpus all -v E:/:/data aegis /bin/bash
```

## Mount Points

When you run the container, your E: drive will be accessible at `/data` inside the container:

- **Host E: drive** → **Container `/data`**
- **Host `./results`** → **Container `/app/results`**
- **Host `./Data`** → **Container `/app/Data`**

## Example Usage

Once inside the container, you can access your E: drive data:

```bash
# List contents of E: drive
ls /data

# Access specific files
cat /data/myfile.txt

# Run AEGIS commands with data from E: drive
aegis --data_root_dir /data/features_uni_v2 ...
```

## Building the Image

The Dockerfile uses `uv` for fast package installation. To build:

```bash
docker build -t aegis .
```

**What the Dockerfile does:**
1. Uses NVIDIA PyTorch base image (already includes PyTorch, torchvision, torchaudio)
2. Installs `uv` package manager
3. Filters requirements.txt to exclude PyTorch packages (already in base image)
4. Installs all other dependencies using `uv pip install --system`
5. Installs the aegis package itself

**Build with cache:**
```bash
docker build -t aegis .
```

**Build without cache (fresh build):**
```bash
docker build --no-cache -t aegis .
```

## Troubleshooting

### Permission Issues
If you encounter permission issues accessing files on E: drive:
- Ensure Docker has access to the E: drive in Docker Desktop settings
- On WSL, you may need to mount the drive: `sudo mount -t drvfs E: /mnt/e`

### GPU Not Available
Make sure you have:
- NVIDIA Docker runtime installed
- GPU drivers installed
- Docker Desktop with GPU support enabled (if using Docker Desktop)

### Path Issues
- **WSL**: Use `/mnt/e` format
- **Windows Native Docker**: Use `E:/` format
- The helper scripts auto-detect your environment

## Customizing Mount Points

Edit `docker-compose.yml` or the helper scripts to change mount points:

```yaml
volumes:
  - /mnt/e:/data:rw  # Change /data to your preferred path
  - ./results:/app/results:rw
```

