@echo off
rem Double-click to install the Membrane Neuron Simulator: creates a Python environment in
rem %LOCALAPPDATA%\MembraneNeuronSimulator\venv (a short path: Windows limits paths to 260 characters) and installs
rem the packages of requirements.txt into it. Run it again to repair or update the packages.
rem Needs Python 3.10, 3.11 or 3.12 (Anaconda, Miniconda or python.org) and an internet connection.
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\MembraneNeuronSimulator\venv"

set "PY="
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%ProgramData%\anaconda3\python.exe" "%ProgramData%\miniconda3\python.exe") do (
    if not defined PY if exist %%P set "PY=%%~P"
)
if not defined PY (
    for %%V in (3.11 3.12 3.10) do (
        if not defined PY py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "pass" >nul 2>&1 && set "PY=python"
)
if not defined PY goto no_python

%PY% -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 (
    echo The Python found ^(%PY%^) is not version 3.10 to 3.12.
    goto no_python
)
echo Using Python: %PY%

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating the environment in %VENV% ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto failed
)
echo Installing the packages ^(the first time downloads about 1 GB; this takes a few minutes^) ...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
"%VENV%\Scripts\python.exe" -c "import numpy, scipy, torch, gmsh, PyQt5, pyvista, pyvistaqt, matplotlib, yaml; import membrane_sim, app, mns_api"
if errorlevel 1 goto failed

echo.
echo Installed. Start the program with "Membrane Neuron Simulator.bat".
echo Optional: MiKTeX ^(pdflatex^) for the PDF reports, Claude Code for the agentic research study.
pause
exit /b 0

:no_python
echo.
echo No suitable Python found. Install Anaconda ^(https://www.anaconda.com/download^) or Python 3.11 from
echo https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run install.bat again.
pause
exit /b 1

:failed
echo.
echo The installation failed; see the messages above. Check the internet connection and run install.bat again.
pause
exit /b 1
