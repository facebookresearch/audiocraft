@echo off
set PATH=.\venv\bin;%PATH%

:activate_venv
set PYTHON="venv\Scripts\Python.exe"
echo venv %PYTHON%

%PYTHON% demos\musicgen_app.py
