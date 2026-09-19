@echo off
setlocal
cd /d "%~dp0"

set "PYMOLAI_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYMOLAI_PYTHON%" (
  echo PyMolAI venv not found: "%PYMOLAI_PYTHON%"
  echo Create it with: uv venv .venv --python 3.12
  pause
  exit /b 1
)

echo Launching PyMolAI...
"%PYMOLAI_PYTHON%" "%~dp0launch_pymolai.py" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo PyMolAI exited with code %EXITCODE%.
  pause
)
exit /b %EXITCODE%
