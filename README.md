# Video Rename Tool

AI-powered video organization with metadata backup/restore.

## Features
- **Batch process videos in a directory**
- **Compress videos to 360p proxy (<20MB) using FFmpeg**
- **Analyze video content with Google Gemini AI**
- **Get suggested filename, description, and tags**
- **Update video metadata and rename files**
- **Create timestamped JSON backups for easy restore**
- **Restore original filenames and metadata from backup**
- **Optional parallel processing for speed**

## Requirements
- Python 3.9+
- FFmpeg installed (`brew install ffmpeg`)
- Google Gemini API key (set as `GOOGLE_API_KEY` in `.env`)
- Python packages: `google-genai`, `pydantic`, `python-dotenv`

## Setup
1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your-gemini-api-key-here
   ```
3. Ensure FFmpeg is installed and available in your PATH.

## Usage

### Process Videos
```bash
python video_rename.py /path/to/videos
```

### Preview Changes (Dry Run)
```bash
python video_rename.py /path/to/videos --dry-run
```

### Parallel Processing
```bash
python video_rename.py /path/to/videos --parallel --workers 4
```

### Restore from Backup
```bash
python video_rename.py --restore video_rename_backup_YYYYMMDD_HHMMSS.json
```

### Other Options
- `--output-dir DIR` : Output directory for renamed videos
- `--extensions EXT ...` : File extensions to process (default: .mp4 .mov .avi .mkv .webm .m4v)
- `--max-proxy-size MB` : Maximum proxy file size (default: 20)

## How It Works
1. **Compression**: Each video is compressed to a 360p proxy using FFmpeg, targeting <20MB.
2. **AI Analysis**: Proxy is uploaded to Gemini AI, which returns a descriptive filename, summary, and tags.
3. **Metadata Update**: The original video is renamed and metadata updated using FFmpeg.
4. **Backup**: A JSON log is created for every run, storing original and new metadata/filenames.
5. **Restore**: You can revert all changes using the backup JSON.

## Notes
- The tool loads your API key from `.env` automatically.
- If a video is still processing on Gemini, the script will wait until it's ready.
- All changes are logged for easy restoration.

## License
MIT
