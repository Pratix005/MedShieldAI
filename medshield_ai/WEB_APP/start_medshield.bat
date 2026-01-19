@echo off
title 🚀 MedShield AI Launcher
echo ============================================
echo  Starting MedShield AI Web Application...
echo  Please wait while the environment loads.
echo ============================================

REM --- Activate conda environment ---
CALL C:\Users\HP\anaconda3\Scripts\activate.bat medshield_ai

REM --- Go to your Flask app directory ---
cd /d "C:\Users\HP\OneDrive\Desktop\MedShieldAI\medshield_ai\WEB_APP"

REM --- Start the Flask app (unbuffered for live logs) ---
start "" http://127.0.0.1:5000
python -u app.py

echo.
echo ✅ MedShield AI server stopped.
pause
