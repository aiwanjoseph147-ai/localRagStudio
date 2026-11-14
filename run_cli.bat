@echo off
echo Starting RAG Document QA CLI...
if exist "..\.venv\Scripts\python.exe" (
    ..\.venv\Scripts\python.exe cli.py %*
) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe cli.py %*
) else (
    python cli.py %*
)
pause
