@echo off
REM Face Recognition System - Build Script (Windows)

echo ==================================
echo Building Face Recognition System
echo ==================================

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not installed
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist output mkdir output
if not exist models mkdir models
if not exist config mkdir config

REM Copy environment file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file with your configuration
)

REM Build Docker image
echo Building Docker image...
docker-compose build

echo ==================================
echo Build complete!
echo ==================================
echo.
echo Next steps:
echo 1. Edit .env file with your RTSP URL and settings
echo 2. Run: scripts\run.bat
echo.

pause
