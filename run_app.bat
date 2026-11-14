@echo off
echo Starting RAG Studio Web Application...
if exist "..\.venv\Scripts\python.exe" (
    ..\.venv\Scripts\python.exe app.py
) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
