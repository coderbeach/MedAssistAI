@echo off
echo Starting Streamlit App inside local Virtual Environment...
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py
pause
