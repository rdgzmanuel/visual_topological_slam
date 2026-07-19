@echo off
REM One-command launcher for Windows (requires Docker Desktop).
cd /d "%~dp0"
docker compose run --rm prism
if errorlevel 1 (
    echo.
    echo GPU run failed to start. Falling back to CPU...
    docker compose run --rm prism-cpu
)
pause
