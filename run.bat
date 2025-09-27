@echo off
setlocal enabledelayedexpansion

REM Configuration
set CONTAINER_NAME=webcast-container
set IMAGE_NAME=webcast
set PORT_MAP=8501:8501
set ENV_FILE=.env
set HOST_DIR=%cd%
set CONTAINER_DIR=/app

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed or not in PATH.
    exit /b 1
)

REM Check if the image exists
docker image inspect %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Image '%IMAGE_NAME%' not found. Building...
    docker build -t %IMAGE_NAME% .
)

REM Handle .env file
set ENV_ARGS=
if exist %ENV_FILE% (
    set ENV_ARGS=--env-file %ENV_FILE%
) else (
    echo Warning: %ENV_FILE% not found; continuing without --env-file.
)

REM Stop running container if exists
docker ps --format "{{.Names}}" | findstr /R /C:"^%CONTAINER_NAME%$" >nul
if %errorlevel%==0 (
    echo Stopping running container: %CONTAINER_NAME%
    docker stop %CONTAINER_NAME% >nul
)

REM Remove container if exists
docker ps -a --format "{{.Names}}" | findstr /R /C:"^%CONTAINER_NAME%$" >nul
if %errorlevel%==0 (
    docker rm %CONTAINER_NAME% >nul
)

REM Run the container
docker run --rm ^
    --name %CONTAINER_NAME% ^
    %ENV_ARGS% ^
    -p %PORT_MAP% ^
    -v "%HOST_DIR%:%CONTAINER_DIR%" ^
    %IMAGE_NAME%

endlocal
pause
