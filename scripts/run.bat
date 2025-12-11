@echo off
REM Face Recognition System - Run Script (Windows)

echo ==================================
echo Starting Face Recognition System
echo ==================================

REM Check if .env exists
if not exist .env (
    echo Error: .env file not found
    echo Please run scripts\build.bat first
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not running
    echo Please start Docker Desktop
    exit /b 1
)

REM Start the system
echo Starting containers...
docker-compose up

echo ==================================
echo System stopped
echo ==================================

pause
