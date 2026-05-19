@echo off
REM Helper script to run Docker container with E: drive mounted (Windows batch file)
REM Usage: run_docker.bat [command]

setlocal

REM Check if Docker image exists
docker images | findstr /C:"aegis" >nul
if errorlevel 1 (
    echo Building Docker image...
    docker build -t aegis .
)

REM Default command (interactive bash shell)
if "%1"=="" (
    set CMD=/bin/bash
) else (
    set CMD=%*
)

REM Run the container with E: drive mounted
echo Running Docker container with E: drive mounted at /data
echo Command: %CMD%
docker run -it --rm ^
    --gpus all ^
    --shm-size 512g ^
    --memory 96g ^
    --cpus 32 ^
    -v E:/:/data:rw ^
    -v %CD%\results:/app/results:rw ^
    -v %CD%\Data:/app/Data:rw ^
    -w /app ^
    aegis ^
    %CMD%

endlocal

