@echo off
rem Double-click to start the Membrane Neuron Simulator: uses the environment made by install.bat, otherwise the
rem Anaconda Python.
cd /d "%~dp0"
set "PYW=%LOCALAPPDATA%\MembraneNeuronSimulator\venv\Scripts\pythonw.exe"
if not exist "%PYW%" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" run_app.py %*
