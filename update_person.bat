@echo off
REM Update Person Embeddings Batch Script
REM Improves face recognition accuracy

cd /d "%~dp0"

REM Check if Docker Desktop is running
"C:\Program Files\Docker\Docker\resources\bin\docker.exe" ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop and wait for it to be ready.
    pause
    exit /b 1
)

REM Run the update script
python update_person_embeddings.py

pause
