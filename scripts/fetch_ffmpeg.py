#!/usr/bin/env python3
"""
Download pre-built static FFmpeg binaries into bin/ for the current platform.

Usage:
    python scripts/fetch_ffmpeg.py

After running, bin/ will contain ffmpeg + ffprobe (or ffmpeg.exe + ffprobe.exe
on Windows).  The PyInstaller build then bundles them inside the app.

If you already have ffmpeg installed system-wide you can skip this and the app
will fall back to the system PATH version at runtime.
"""

import platform
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

BIN_DIR = Path(__file__).parent.parent / 'bin'

# BtbN builds static GPL FFmpeg for Linux and Windows.
# macOS users should install via Homebrew; the CI workflow does this too.
SOURCES = {
    'linux': {
        'url': (
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/'
            'ffmpeg-master-latest-linux64-gpl.tar.xz'
        ),
        'format': 'tar',
        'prefix': 'ffmpeg-master-latest-linux64-gpl/bin/',
        'files': ['ffmpeg', 'ffprobe'],
    },
    'windows': {
        'url': (
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/'
            'ffmpeg-master-latest-win64-gpl.zip'
        ),
        'format': 'zip',
        'prefix': 'ffmpeg-master-latest-win64-gpl/bin/',
        'files': ['ffmpeg.exe', 'ffprobe.exe'],
    },
}


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 // total_size, 100)
        print(f'\r  {pct}%', end='', flush=True)


def main():
    system = platform.system().lower()
    if system == 'darwin':
        print('macOS: install FFmpeg via Homebrew, then copy the binaries:')
        print()
        print('  brew install ffmpeg')
        print('  cp $(brew --prefix)/bin/ffmpeg  bin/ffmpeg')
        print('  cp $(brew --prefix)/bin/ffprobe bin/ffprobe')
        print()
        print('The GitHub Actions CI workflow does this automatically.')
        sys.exit(0)

    os_key = 'windows' if system == 'windows' else 'linux'
    cfg = SOURCES[os_key]

    BIN_DIR.mkdir(exist_ok=True)

    archive_name = cfg['url'].rsplit('/', 1)[-1]
    archive_path = BIN_DIR / archive_name

    print(f'Downloading {archive_name} ...')
    urllib.request.urlretrieve(cfg['url'], archive_path, reporthook=_reporthook)
    print()

    print('Extracting binaries ...')
    if cfg['format'] == 'tar':
        with tarfile.open(archive_path) as tar:
            for fname in cfg['files']:
                member_path = cfg['prefix'] + fname
                member = tar.getmember(member_path)
                member.name = fname          # strip the path prefix
                tar.extract(member, BIN_DIR)
    else:
        with zipfile.ZipFile(archive_path) as zf:
            for fname in cfg['files']:
                data = zf.read(cfg['prefix'] + fname)
                dest = BIN_DIR / fname
                dest.write_bytes(data)
                if os_key != 'windows':
                    dest.chmod(0o755)

    archive_path.unlink()

    print(f'Done. Binaries installed to {BIN_DIR}/:')
    for f in sorted(BIN_DIR.iterdir()):
        if not f.name.startswith('.'):
            print(f'  {f.name}')


if __name__ == '__main__':
    main()
