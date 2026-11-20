@echo off
title Crop Disease Detection - REST API
echo ========================================================
echo Starting Crop Disease Detection REST API (FastAPI)
echo Docs available at: http://localhost:8000/docs
echo ========================================================
echo.

IF EXIST ".\venv\Scripts\python.exe" (
    echo Using project virtual environment...
    .\venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000 --reload
) ELSE (
    echo Virtual environment not found. Using system Python...
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
)

pause
