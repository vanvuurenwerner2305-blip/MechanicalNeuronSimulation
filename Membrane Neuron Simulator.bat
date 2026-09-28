@echo off
rem Double-click to start the Membrane Neuron Simulator (uses the Anaconda Python).
cd /d "%~dp0"
set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" run_app.py %*
