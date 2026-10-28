@echo off
REM iCreate Music - one-command start (Windows)
cd /d "%~dp0"

python -c "import numpy" 2>NUL
if errorlevel 1 (
  echo Installing numpy ^(one time^)...
  python -m pip install --quiet numpy
)

python -m icreate_music %*
pause
