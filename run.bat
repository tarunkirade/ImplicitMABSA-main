@echo off
setlocal
cd /d "%~dp0"
title ImplicitMABSA
set "PYTHONPATH=%~dp0src"
set "PYTHONUTF8=1"

rem ---------- find a real Python (the py launcher first; plain python only if it has pip) ----------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY ( python -c "import pip" >nul 2>&1 && set "PY=python" )
if not defined PY goto nopython

rem ---------- first-time setup (skipped automatically when already done) ----------
%PY% -c "import torch, PIL, numpy, open_clip, vaderSentiment" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  First run: installing torch, numpy, Pillow, open_clip and vaderSentiment - a few minutes ...
    %PY% -m pip install -r requirements-ui.txt
    if errorlevel 1 goto pipfail
)
if not exist "data\images\train_00000.png" (
    echo.
    echo  First run: creating the dataset - about 1 minute ...
    %PY% run.py setup
    if errorlevel 1 goto fail
)

:menu
cls
echo.
echo  =====================================================
echo    ImplicitMABSA  -  text + image  -  topic + feeling
echo  =====================================================
echo.
echo    [1]  Open the app in your browser        (optional UI)
echo    [2]  Code only: analyse a random example
echo    [3]  Code only: measure the model (accuracy, precision, recall, F1)
echo    [4]  Code only: show the real-data (Twitter) results
echo    [5]  Re-create the dataset
echo    [Q]  Quit
echo.
choice /c 12345Q /n /m "  Your choice: "
if errorlevel 6 goto end
if errorlevel 5 goto do5
if errorlevel 4 goto do4
if errorlevel 3 goto do3
if errorlevel 2 goto do2
goto do1

:do1
echo.
echo  Starting the app - keep this window open.
echo  To stop it and come back to this menu: click "Quit app" (top right of the page).
echo  If you press Ctrl+C instead and Windows asks "Terminate batch job (Y/N)?", answer N.
%PY% run.py ui
goto again
:do2
echo.
%PY% run.py demo
goto again
:do3
echo.
%PY% run.py evaluate --split test_iid
%PY% run.py evaluate --split test_ood
goto again
:do4
%PY% run.py real
goto again
:do5
%PY% run.py setup --force
goto again

:again
echo.
pause
goto menu

:nopython
echo.
echo  No usable Python found.
echo  Install Python 3.11 or 3.12 from https://www.python.org/downloads/
echo  and tick "Add python.exe to PATH" in the installer, then run this file again.
goto end
:pipfail
echo.
echo  Package installation failed. Check your internet connection and run again.
goto end
:fail
echo.
echo  Something failed - see the message above.
:end
echo.
pause
