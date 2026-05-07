#!/usr/bin/env bash
# Build the Smart Video Rename desktop app for macOS or Linux.
# Run from the repo root or from the scripts/ directory.
set -euo pipefail
cd "$(dirname "$0")/.."

# ── FFmpeg binaries ───────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ ! -f bin/ffmpeg ]; then
        echo "Fetching FFmpeg via Homebrew..."
        brew install ffmpeg
        cp "$(brew --prefix)/bin/ffmpeg"  bin/ffmpeg
        cp "$(brew --prefix)/bin/ffprobe" bin/ffprobe
    fi
else
    if [ ! -f bin/ffmpeg ]; then
        echo "Fetching FFmpeg static build..."
        python scripts/fetch_ffmpeg.py
    fi
fi

# ── PyInstaller ───────────────────────────────────────────────────────────────
echo "Running PyInstaller..."
pyinstaller smart_video_rename.spec --clean

# ── Platform installer ────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Creating .dmg..."
    hdiutil create \
        -volname "Smart Video Rename" \
        -srcfolder "dist/SmartVideoRename.app" \
        -ov -format UDZO \
        "dist/SmartVideoRename-macOS.dmg"
    echo "→ dist/SmartVideoRename-macOS.dmg"

else
    # Linux — create an AppImage if appimagetool is available
    APPDIR="dist/SmartVideoRename.AppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin"
    cp -r dist/SmartVideoRename/* "$APPDIR/usr/bin/"

    cat > "$APPDIR/SmartVideoRename.desktop" <<'EOF'
[Desktop Entry]
Name=Smart Video Rename
Exec=SmartVideoRename
Icon=SmartVideoRename
Type=Application
Categories=Utility;AudioVideo;
EOF

    cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
exec "$(dirname "$0")/usr/bin/SmartVideoRename" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    if command -v appimagetool &>/dev/null; then
        appimagetool "$APPDIR" "dist/SmartVideoRename-Linux.AppImage"
        echo "→ dist/SmartVideoRename-Linux.AppImage"
    else
        echo "appimagetool not found — distributable folder: dist/SmartVideoRename/"
        echo "Install appimagetool from https://github.com/AppImage/AppImageKit/releases"
    fi
fi
