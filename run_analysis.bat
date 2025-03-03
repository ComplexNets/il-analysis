@echo off
echo Checking Python environment...

REM Activate conda environment using the proper Conda activation script
call C:\Users\X1\miniconda3\Scripts\activate.bat ils 2>nul
if %errorlevel% equ 0 (
    echo Successfully activated 'ils' conda environment
) else (
    echo Warning: Could not activate 'ils' conda environment using standard path
    echo Trying alternative activation method...
    call activate ils 2>nul
    if %errorlevel% equ 0 (
        echo Successfully activated 'ils' conda environment
    ) else (
        echo Failed to activate conda environment.
        echo Please make sure the 'ils' environment exists and try again.
        pause
        exit /b 1
    )
)

echo Starting ILS Analysis Tool...
cd %~dp0
echo Current directory: %CD%
echo.
echo *** IMPORTANT: DO NOT CLOSE THIS WINDOW ***
echo * Keep this window open while using the Analysis Tool
echo * To stop the Analysis Tool, press Ctrl+C in this window, then type 'Y' to terminate
echo * Closing this window will terminate the Analysis Tool
echo.

REM Set initial port
set PORT=8504

REM Check if port is in use and increment until an available port is found
:CHECK_PORT
powershell -Command "if ((Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue).Count -gt 0) { exit 1 } else { exit 0 }"
if %errorlevel% equ 1 (
    echo Port %PORT% is in use. Trying port %PORT%+1...
    set /a PORT+=1
    goto CHECK_PORT
)

echo Using port %PORT%

REM Run streamlit without pause - the command window will stay open while streamlit runs
streamlit run frontend/new_analysis_page.py --server.port=%PORT%

REM This point is only reached when streamlit is closed
echo.
echo Analysis Tool has been closed.
echo Press any key to exit...
pause
