@echo off
REM Build the Smart Video Rename desktop app for Windows.
REM Run from the repo root or the scripts\ directory.
setlocal

cd /d "%~dp0\.."

REM ── FFmpeg binaries ──────────────────────────────────────────────────────────
if not exist bin\ffmpeg.exe (
    echo Fetching FFmpeg static build...
    python scripts\fetch_ffmpeg.py
)

REM ── PyInstaller ───────────────────────────────────────────────────────────────
echo Running PyInstaller...
pyinstaller smart_video_rename.spec --clean
if errorlevel 1 (
    echo PyInstaller failed.
    exit /b 1
)

echo.
echo Build complete.
echo Distributable folder: dist\SmartVideoRename\
echo.
echo To create a .exe installer install Inno Setup, then run:
echo   iscc scripts\installer.iss
