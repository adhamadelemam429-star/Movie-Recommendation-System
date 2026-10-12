@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" (
    .\.venv\Scripts\streamlit.exe run app/ui.py
) else (
    echo Streamlit environment not found. Please create the virtual environment first.
    pause
)
