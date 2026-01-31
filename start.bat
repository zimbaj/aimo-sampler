@echo off
title AIMO Sampler - AI Audio Generator
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           AIMO SAMPLER - AI Audio Generator               ║
echo  ║              Powered by Meta's MusicGen                   ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Check if venv exists
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create it first with:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo [*] Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

:: Change to project directory
cd /d "%~dp0"

echo [*] Environment ready!
echo.
echo  ┌─────────────────────────────────────────────────────────┐
echo  │                    QUICK COMMANDS                       │
echo  ├─────────────────────────────────────────────────────────┤
echo  │  Generate samples:                                      │
echo  │    python -m src.cli generate "your prompt here"        │
echo  │                                                         │
echo  │  List presets:                                          │
echo  │    python -m src.cli presets                            │
echo  │                                                         │
echo  │  Use preset:                                            │
echo  │    python -m src.cli generate --preset techno-kick      │
echo  │                                                         │
echo  │  Show help:                                             │
echo  │    python -m src.cli --help                             │
echo  └─────────────────────────────────────────────────────────┘
echo.

:menu
echo.
echo  [1] Generate sample (interactive)
echo  [2] List presets
echo  [3] Show available models
echo  [4] Open command prompt (manual mode)
echo  [5] Exit
echo.
set /p choice="  Select option (1-5): "

if "%choice%"=="1" goto generate
if "%choice%"=="2" goto presets
if "%choice%"=="3" goto models
if "%choice%"=="4" goto cmdmode
if "%choice%"=="5" goto end

echo Invalid option. Please try again.
goto menu

:generate
echo.
set /p prompt="  Enter your prompt: "
set /p duration="  Duration in seconds (default=5): "
set /p variations="  Number of variations (default=1): "

if "%duration%"=="" set duration=5
if "%variations%"=="" set variations=1

echo.
echo [*] Generating audio...
python -m src.cli generate "%prompt%" --duration %duration% --variations %variations% --precise
echo.
goto menu

:presets
echo.
python -m src.cli presets
goto menu

:models
echo.
python -m src.cli models
goto menu

:cmdmode
echo.
echo [*] Entering manual mode. Type 'exit' to return to menu.
echo.
cmd /k "echo AIMO Sampler Ready! && echo."

:end
echo.
echo Goodbye!
pause
