@echo off
echo Starting Jupyter Notebook inside local Virtual Environment...
cd /d "%~dp0"
.venv\Scripts\python.exe -m jupyter notebook
pause
