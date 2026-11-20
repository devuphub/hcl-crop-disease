@echo off
title Crop Disease Detection - Web Application
echo ========================================================
echo Starting Crop Disease Detection System (Streamlit App)
echo ========================================================
echo.

IF EXIST ".\venv\Scripts\python.exe" (
    echo Using project virtual environment...
    .\venv\Scripts\streamlit.exe run web\app.py
) ELSE (
    echo Virtual environment not found. Using system Python...
    streamlit run web\app.py
)

pause
