@echo off
echo ========================================
echo     Starting CMS with Conda Environment
echo ========================================
echo.

:: Change to the directory where the script is located (in case you double-click)
cd /d "%~dp0"

:: Activate Conda environment
echo Activating Conda environment...

:: Replace "aiml-cset312" with your actual conda environment name
call conda activate aiml-cset312

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate conda environment "aiml-cset312"
    echo Please check if the environment name is correct.
    pause
    exit /b 1
)

echo Conda environment activated successfully.
echo.

:: Run the Uvicorn server
echo Starting FastAPI server...
echo Running: uvicorn main:app --reload --port 8000
echo Server URL: http://127.0.0.1:8000
echo Press Ctrl + C to stop the server.
echo.

python -m uvicorn main:app --reload --port 8000

echo.
echo Server stopped.
pause