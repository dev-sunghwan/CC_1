@echo off
REM Face Registration Batch Script
REM Works without Docker in PATH

cd /d "%~dp0"

REM Check if Docker Desktop is running
"C:\Program Files\Docker\Docker\resources\bin\docker.exe" ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop and wait for it to be ready.
    pause
    exit /b 1
)

REM Run the registration script
python register_new_person.py

pause
