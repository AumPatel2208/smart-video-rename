# Video Rename Tool

AI-powered video organization with metadata backup/restore.

> Note: This is a vibe coded project that's been tested and is working on [Amenno's](https://www.amenno.co.uk) pipeline.


## Features
- **Batch process videos in a directory**
- **Creates 360p proxy (<20MB) using FFmpeg to upload to AI**
- **Analyze video content with Google Gemini AI**

- **Rename with suggested filename + metadata description, and tags**
- **Update video metadata and rename files**
- **Create timestamped JSON backups for easy restore**
- **Restore original filenames and metadata from backup**
- **Optional parallel processing for speed**

## Requirements
- Python 3.9+
- FFmpeg installed (`brew install ffmpeg`)
- Google Gemini API key (set as `GOOGLE_API_KEY` in `.env`)
- Python packages: `google-genai`, `pydantic`, `python-dotenv`


# Video Rename Tool

AI-powered video organization and metadata backup/restore.

## Features

- **Proxy uploads:** 
  - Automatically generates 360p proxy files under a configurable size limit (default 20MB) to upload to Gemini.
  - For image+text (no-audio) models, i.e. Gemma, it extracts 32frames from video and uses whisper for transcription to provide metadata
- **AI Metadata Analysis:** Uses Google Gemini or Gemma models to analyze video content and suggest descriptive filenames, concise descriptions, and relevant tags.
- **Slate/Clapperboard Detection: (UNTESTED)** Optionally detects Scene, Shot, and Take information from visible slates/clapperboards in the video.
- **Metadata Writing:** Updates video files with new metadata (title, description, tags) using ffmpeg (no re-encoding).
- **Backup & Restore:** Creates a JSON backup log for all processed videos, allowing full restore of filenames and metadata.
- **DaVinci Resolve CSV Export:** Generates a CSV file compatible with DaVinci Resolve for easy import of metadata.
- **Parallel Processing:** Supports multi-threaded video processing for faster batch operations.
- **Dry Run Mode:** Preview all changes (renames, metadata updates) without modifying any files.

## Usage

```bash
python video_rename.py /path/to/videos
```

### Options

- `--dry-run` / `-n`: Preview changes without applying them
- `--parallel` / `-p`: Enable parallel processing of videos
- `--workers` / `-w`: Number of parallel workers (default: 4)
- `--detect-slate`: Enable slate/clapperboard detection for Scene, Shot, Take fields
- `--output-dir` / `-o`: Output directory for renamed videos (default: same as input)
- `--extensions` / `-e`: Video file extensions to process (default: .mp4 .mov .avi .mkv .webm .m4v)
- `--max-proxy-size` / `-s`: Maximum proxy file size in MB (default: 20)
- `--model` / `-m`: AI model to use (`gemini-2.5-flash` or `gemma-3-27b-it`)
- `--restore` / `-r <backup.json>`: Restore videos from a backup JSON file

### Output

- JSON backup file for restore capability
- DaVinci Resolve compatible CSV with metadata

## Requirements

- Python 3.9+
- ffmpeg, ffprobe
- Google Gemini API credentials
- whisper, pillow, pydantic, python-dotenv

## Example

```bash
python video_rename.py /path/to/videos --dry-run --parallel --detect-slate
```

## Restore

```bash
python video_rename.py --restore backup.json
```
- All changes are logged for easy restoration.

## License
MIT
