# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Smart Video Rename."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# NiceGUI ships its own web assets (JS/CSS/HTML) that must travel with the exe.
nicegui_datas = collect_data_files('nicegui')
nicegui_hidden = collect_submodules('nicegui')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=nicegui_datas + [
        # Vendored ffmpeg/ffprobe binaries — populated by scripts/fetch_ffmpeg.py
        ('bin', 'bin'),
    ],
    hiddenimports=nicegui_hidden + [
        'video_rename',
        'google.genai',
        'pydantic',
        'dotenv',
        'whisper',
        'PIL',
        'tkinter',
        'pywebview',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SmartVideoRename',
    debug=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window shown to end users
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='SmartVideoRename',
)

# macOS: wrap the collected folder in a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='SmartVideoRename.app',
        bundle_identifier='com.smartvideorename.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.14',
            'CFBundleDisplayName': 'Smart Video Rename',
            'CFBundleShortVersionString': '1.0.0',
        },
    )
