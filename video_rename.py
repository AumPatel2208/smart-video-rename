#!/usr/bin/env python3
"""
Video Rename Tool - AI-powered video organization with metadata backup/restore.

This tool processes videos in a directory:
1. Creates 360p proxy files under 20MB
2. Uploads to Google Gemini for AI analysis
3. Gets suggested filename, description, and tags
4. Updates video metadata and renames files
5. Creates JSON backup for restore capability
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time
from typing import Callable, List, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

# Routing all output through _log lets callers (GUI or CLI) capture progress messages.
# GUI sets this to its own callback via run_processing(on_progress=...); CLI keeps print.
_log: Callable = print


def _ffmpeg_bin(name: str) -> str:
    """Return the path to an ffmpeg binary.

    When running inside a PyInstaller bundle the vendored binaries live in
    sys._MEIPASS/bin/.  Otherwise we fall back to whatever is on PATH.
    """
    if getattr(sys, 'frozen', False):
        bundled = Path(sys._MEIPASS) / 'bin' / name  # type: ignore[attr-defined]
        if bundled.exists():
            return str(bundled)
    return name  # rely on PATH


# ============ Data Models ============

class VideoMetadata(BaseModel):
    """Structured metadata from AI analysis."""
    filename: str = Field(
        description="Descriptive filename (lowercase, underscores instead of spaces, "
                    "max 50 chars, no extension, filesystem-safe characters only)"
    )
    description: str = Field(
        description="Concise 1-2 sentence description of the video content"
    )
    tags: List[str] = Field(
        description="3-7 relevant keyword tags for categorization"
    )
    # Slate detection fields (only populated when --detect-slate is used)
    scene: Optional[str] = Field(
        default=None,
        description="Scene number/identifier from slate/clapperboard if visible"
    )
    shot: Optional[str] = Field(
        default=None,
        description="Shot number/identifier from slate/clapperboard if visible"
    )
    take: Optional[str] = Field(
        default=None,
        description="Take number from slate/clapperboard if visible"
    )


class OriginalMetadata(BaseModel):
    """Original video metadata for backup/restore."""
    title: Optional[str] = None
    description: Optional[str] = None
    comment: Optional[str] = None
    keywords: Optional[str] = None


class VideoBackupEntry(BaseModel):
    """Backup entry for a single video."""
    original_path: str
    original_filename: str
    new_path: str
    new_filename: str
    original_metadata: OriginalMetadata
    new_metadata: VideoMetadata
    processed_at: str


class BackupLog(BaseModel):
    """Complete backup log for a processing session."""
    created_at: str
    source_directory: str
    detect_slate: bool = False
    entries: List[VideoBackupEntry] = []


# ============ FFmpeg Functions ============

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [_ffmpeg_bin('ffprobe'), '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def get_video_metadata(video_path: str) -> OriginalMetadata:
    """Extract existing metadata from video using ffprobe."""
    result = subprocess.run(
        [_ffmpeg_bin('ffprobe'), '-v', 'error', '-show_entries',
         'format_tags=title,description,comment,keywords',
         '-of', 'json', video_path],
        capture_output=True, text=True
    )

    metadata = OriginalMetadata()

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            tags = data.get('format', {}).get('tags', {})
            # Handle case-insensitive tag keys
            tags_lower = {k.lower(): v for k, v in tags.items()}
            metadata.title = tags_lower.get('title')
            metadata.description = tags_lower.get('description')
            metadata.comment = tags_lower.get('comment')
            metadata.keywords = tags_lower.get('keywords')
        except (json.JSONDecodeError, KeyError):
            pass

    return metadata


def create_proxy(input_path: str, output_path: str, max_size_mb: float = 20.0) -> bool:
    """Create 360p proxy video under size limit."""
    try:
        duration = get_video_duration(input_path)
    except subprocess.CalledProcessError:
        _log(f"    Warning: Could not get duration, using default bitrate")
        duration = 300  # Assume 5 minutes if we can't get duration

    # Calculate bitrate for target size (leave room for audio)
    audio_bitrate = 64  # kbps
    target_bitrate = int((max_size_mb * 8 * 1024) / duration) - audio_bitrate
    target_bitrate = max(target_bitrate, 100)  # Minimum bitrate

    # FFmpeg command breakdown:
    #   -y                    : Overwrite output file without asking
    #   -i input_path         : Input video file
    #   -vf scale=-2:360      : Scale to 360p height, auto-calculate width (divisible by 2)
    #   -c:v libx264          : Use H.264 video codec
    #   -preset fast          : Encoding speed/compression tradeoff (fast = quicker, larger file)
    #   -b:v {bitrate}k       : Target video bitrate in kbps (calculated to hit size target)
    #   -c:a aac              : Use AAC audio codec
    #   -b:a 64k              : Audio bitrate 64kbps
    #   -movflags +faststart  : Move metadata to start for faster streaming/upload
    result = subprocess.run([
        _ffmpeg_bin('ffmpeg'), '-y', '-i', input_path,
        '-vf', 'scale=-2:360',
        '-c:v', 'libx264', '-preset', 'fast',
        '-b:v', f'{target_bitrate}k',
        '-c:a', 'aac', '-b:a', f'{audio_bitrate}k',
        '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        _log(f"    FFmpeg error: {result.stderr[:500]}")
        return False

    # Verify file size
    actual_size = os.path.getsize(output_path) / (1024 * 1024)
    if actual_size > max_size_mb:
        _log(f"    Warning: Proxy is {actual_size:.1f}MB (target was {max_size_mb}MB)")

    return True


def write_metadata(input_path: str, output_path: str,
                   title: str, description: str, tags: List[str]) -> bool:
    """Write metadata to video file using ffmpeg (stream copy, no re-encode)."""
    keywords = ', '.join(tags)

    result = subprocess.run([
        _ffmpeg_bin('ffmpeg'), '-y', '-i', input_path,
        '-c', 'copy',
        '-movflags', 'use_metadata_tags',
        '-metadata', f'title={title}',
        '-metadata', f'description={description}',
        '-metadata', f'comment={description}',
        '-metadata', f'keywords={keywords}',
        output_path
    ], capture_output=True, text=True)

    return result.returncode == 0


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video file using ffmpeg."""
    result = subprocess.run([
        _ffmpeg_bin('ffmpeg'), '-y', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format for Whisper
        '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
        '-ac', '1',  # Mono
        output_path
    ], capture_output=True, text=True)
    return result.returncode == 0


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    import whisper

    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path)
    return result["text"]


def extract_frames(video_path: str, output_dir: str, num_frames: int = 32) -> List[str]:
    """Extract evenly spaced frames from video.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        num_frames: Number of frames to extract (default 32)

    Returns:
        List of paths to extracted frame images
    """
    duration = get_video_duration(video_path)

    # Calculate frame interval
    # We want frames at: 0, duration/(n-1), 2*duration/(n-1), ..., duration
    if num_frames <= 1:
        fps_filter = f"fps=1/{duration}"
    else:
        interval = duration / (num_frames - 1)
        fps_filter = f"fps=1/{interval}"

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    result = subprocess.run([
        _ffmpeg_bin('ffmpeg'), '-y', '-i', video_path,
        '-vf', f"{fps_filter},scale=512:-1",  # Scale to 512px width for efficiency
        '-vframes', str(num_frames),
        '-q:v', '2',  # High quality JPEG
        output_pattern
    ], capture_output=True, text=True)

    if result.returncode != 0:
        _log(f"    FFmpeg frame extraction error: {result.stderr[:500]}")
        return []

    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])

    return frame_paths[:num_frames]  # Ensure we don't return more than requested


def restore_metadata(input_path: str, output_path: str,
                     metadata: OriginalMetadata) -> bool:
    """Restore original metadata to video file."""
    cmd = [
        _ffmpeg_bin('ffmpeg'), '-y', '-i', input_path,
        '-c', 'copy',
        '-movflags', 'use_metadata_tags',
    ]

    # Only set metadata fields that were originally present
    if metadata.title:
        cmd.extend(['-metadata', f'title={metadata.title}'])
    else:
        cmd.extend(['-metadata', 'title='])

    if metadata.description:
        cmd.extend(['-metadata', f'description={metadata.description}'])
    else:
        cmd.extend(['-metadata', 'description='])

    if metadata.comment:
        cmd.extend(['-metadata', f'comment={metadata.comment}'])
    else:
        cmd.extend(['-metadata', 'comment='])

    if metadata.keywords:
        cmd.extend(['-metadata', f'keywords={metadata.keywords}'])
    else:
        cmd.extend(['-metadata', 'keywords='])

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ============ AI Analysis ============

def analyze_video(proxy_path: str, client: genai.Client, detect_slate: bool = False) -> VideoMetadata:
    """Analyze video with Gemini and return structured metadata."""
    uploaded_file = client.files.upload(file=proxy_path)

    while uploaded_file.state == "PROCESSING":
        _log(f'  Waiting for video to be processed. {uploaded_file.name}:{uploaded_file.state}')
        time.sleep(5)
        _log('')
        uploaded_file = client.files.get(name=uploaded_file.name)

    prompt = """Analyze this video and provide metadata for organizing it.

Based on the video content, provide:
1. filename: A descriptive, filesystem-safe filename
   - Use lowercase letters, numbers, and underscores only
   - Replace spaces with underscores
   - Maximum 50 characters
   - No file extension
   - Make it descriptive of the content

2. description: A concise 1-2 sentence description of what happens in the video

3. tags: 3-7 relevant keyword tags for categorization
   - Include the main subject/activity
   - Location if identifiable
   - Category (travel, family, sports, tutorial, etc.)
   - Any notable people, objects, or events
"""

    if detect_slate:
        prompt += """
4. scene: Look at the first few seconds of the video for a slate/clapperboard.
   - If visible, extract the Scene number/identifier
   - If no slate is visible, set to null

5. shot: From the slate/clapperboard if visible.
   - Extract the Shot number/identifier
   - If no slate is visible, set to null

6. take: From the slate/clapperboard if visible.
   - Extract the Take number
   - If no slate is visible, set to null
"""

    prompt += "\nRespond with valid JSON only."

    max_retries = 3
    retry_delay = 15  # seconds

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VideoMetadata,
                },
            )

            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            return VideoMetadata.model_validate_json(response.text)

        except Exception as e:
            error_str = str(e)
            if '503' in error_str and 'UNAVAILABLE' in error_str:
                if attempt < max_retries - 1:
                    _log(f"  Model overloaded, waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    _log(f"  Max retries reached. Model still overloaded.")
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception:
                        pass
                    raise
            else:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                raise


def analyze_video_with_frames(video_path: str, client: genai.Client,
                               detect_slate: bool = False,
                               num_frames: int = 32) -> VideoMetadata:
    """Analyze video using extracted frames and audio transcription (for Gemma model).

    This function:
    1. Extracts audio and transcribes it with Whisper
    2. Extracts evenly-spaced frames from the video
    3. Sends frames + transcription to Gemma for analysis

    Args:
        video_path: Path to the video file (can be original or proxy)
        client: Gemini API client
        detect_slate: Whether to detect slate/clapperboard info
        num_frames: Number of frames to extract (default 32, max for Gemma)

    Returns:
        VideoMetadata with AI-generated metadata
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as temp_dir:
        _log("  Extracting audio for transcription...")
        audio_path = os.path.join(temp_dir, "audio.wav")
        transcription = ""

        if extract_audio(video_path, audio_path):
            _log("  Transcribing audio with Whisper...")
            try:
                transcription = transcribe_audio(audio_path)
                if transcription:
                    _log(f"  Transcription: {transcription[:100]}...")
            except Exception as e:
                _log(f"  Warning: Audio transcription failed: {e}")
                transcription = ""
        else:
            _log("  Warning: Could not extract audio (video may be silent)")

        _log(f"  Extracting {num_frames} frames...")
        frame_paths = extract_frames(video_path, temp_dir, num_frames)

        if not frame_paths:
            raise ValueError("Failed to extract frames from video")

        _log(f"  Extracted {len(frame_paths)} frames")

        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            frames.append(img)

        prompt = """Analyze these video frames and audio transcription to provide metadata for organizing this video.

"""
        if transcription:
            prompt += f"""AUDIO TRANSCRIPTION:
{transcription}

"""

        prompt += """Based on the video frames and audio, provide:
1. filename: A descriptive, filesystem-safe filename
   - Use lowercase letters, numbers, and underscores only
   - Replace spaces with underscores
   - Maximum 50 characters
   - No file extension
   - Make it descriptive of the content

2. description: A concise 1-2 sentence description of what happens in the video

3. tags: 3-7 relevant keyword tags for categorization
   - Include the main subject/activity
   - Location if identifiable
   - Category (travel, family, sports, tutorial, etc.)
   - Any notable people, objects, or events
"""

        if detect_slate:
            prompt += """
4. scene: Look at the first few frames for a slate/clapperboard.
   - If visible, extract the Scene number/identifier
   - If no slate is visible, set to null

5. shot: From the slate/clapperboard if visible.
   - Extract the Shot number/identifier
   - If no slate is visible, set to null

6. take: From the slate/clapperboard if visible.
   - Extract the Take number
   - If no slate is visible, set to null
"""

        prompt += "\nRespond with valid JSON only."

        contents = frames + [prompt]

        max_retries = 3
        retry_delay = 15  # seconds

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=contents,
                )
                fixed_response = response.text.replace("```json", "").replace("```", "").strip()
                return VideoMetadata.model_validate_json(fixed_response)

            except Exception as e:
                error_str = str(e)
                if '503' in error_str and 'UNAVAILABLE' in error_str:
                    if attempt < max_retries - 1:
                        _log(f"  Model overloaded, waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        _log(f"  Max retries reached. Model still overloaded.")
                        raise
                else:
                    raise


# ============ Backup/Restore Functions ============

def create_backup_filename(source_dir: Path) -> Path:
    """Create a timestamped backup filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return source_dir / f"video_rename_backup_{timestamp}.json"


def save_backup(backup: BackupLog, backup_path: Path) -> None:
    """Save backup log to JSON file."""
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(backup.model_dump(), f, indent=2, ensure_ascii=False)


def load_backup(backup_path: Path) -> BackupLog:
    """Load backup log from JSON file."""
    with open(backup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return BackupLog.model_validate(data)


def restore_from_backup(backup_path: Path, dry_run: bool = False) -> None:
    """Restore videos to their original state from a backup file."""
    backup = load_backup(backup_path)

    _log(f"Restoring from backup: {backup_path.name}")
    _log(f"Backup created at: {backup.created_at}")
    _log(f"Entries to restore: {len(backup.entries)}")
    _log('')

    for entry in backup.entries:
        new_path = Path(entry.new_path)
        original_path = Path(entry.original_path)

        _log(f"Restoring: {entry.new_filename} -> {entry.original_filename}")

        if not new_path.exists():
            _log(f"  Warning: {new_path} not found, skipping")
            continue

        if dry_run:
            _log(f"  [DRY RUN] Would restore metadata and rename")
            continue

        with tempfile.NamedTemporaryFile(suffix=new_path.suffix, delete=False) as tmp:
            temp_path = tmp.name

        try:
            if restore_metadata(str(new_path), temp_path, entry.original_metadata):
                shutil.move(temp_path, str(original_path))
                if new_path != original_path and new_path.exists():
                    new_path.unlink()
                _log(f"  Restored successfully")
            else:
                _log(f"  Error: Failed to restore metadata")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            _log(f"  Error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    _log("\nRestore complete!")


# ============ DaVinci Resolve CSV Export ============

DAVINCI_CSV_COLUMNS = [
    'File Name',
    'Comments',
    'Shot',
    'Scene',
    'Take',
]


def create_davinci_csv_filename(source_dir: Path) -> Path:
    """Create a timestamped DaVinci CSV filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return source_dir / f"davinci_metadata_{timestamp}.csv"


def format_date_modified(file_path: Path) -> str:
    """Format file modification time in DaVinci Resolve format."""
    try:
        mtime = os.stat(file_path).st_mtime
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%a %b %d %H:%M:%S %Y")
    except (OSError, ValueError):
        return ""


def append_to_davinci_csv(entry: VideoBackupEntry, csv_path: Path, dry_run: bool = False) -> bool:
    """Append a single entry to DaVinci Resolve CSV, creating file if needed."""
    try:
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DAVINCI_CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()

            new_path = Path(entry.new_path)

            tags_str = ', '.join(entry.new_metadata.tags)
            comments = f"Description: {entry.new_metadata.description}\nTags: {tags_str}"

            if not dry_run and new_path.exists():
                date_modified = format_date_modified(new_path)
            else:
                try:
                    dt = datetime.fromisoformat(entry.processed_at)
                    date_modified = dt.strftime("%a %b %d %H:%M:%S %Y")
                except ValueError:
                    date_modified = ""

            row = {
                'File Name': entry.new_filename,
                'Comments': comments,
                'Shot': entry.new_metadata.shot or '',
                'Scene': entry.new_metadata.scene or '',
                'Take': entry.new_metadata.take or '',
            }
            writer.writerow(row)

        return True
    except Exception as e:
        _log(f"  Warning: Failed to update CSV: {e}")
        return False


def export_davinci_csv(backup: BackupLog, csv_path: Path, dry_run: bool = False) -> bool:
    """Export backup entries to DaVinci Resolve compatible CSV."""
    if not backup.entries:
        _log("No entries to export to CSV")
        return False

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DAVINCI_CSV_COLUMNS)
            writer.writeheader()

            for entry in backup.entries:
                new_path = Path(entry.new_path)

                tags_str = ', '.join(entry.new_metadata.tags)
                comments = f"Description: {entry.new_metadata.description}\nTags: {tags_str}"

                if not dry_run and new_path.exists():
                    date_modified = format_date_modified(new_path)
                else:
                    try:
                        dt = datetime.fromisoformat(entry.processed_at)
                        date_modified = dt.strftime("%a %b %d %H:%M:%S %Y")
                    except ValueError:
                        date_modified = ""

                row = {
                    'File Name': entry.new_filename,
                    'Comments': comments,
                    'Shot': entry.new_metadata.shot or '',
                    'Scene': entry.new_metadata.scene or '',
                    'Take': entry.new_metadata.take or '',
                }
                writer.writerow(row)

        return True
    except Exception as e:
        _log(f"Error exporting CSV: {e}")
        return False


# ============ Video Processing ============

def process_single_video(video_path: Path, output_dir: Path,
                         client: genai.Client, dry_run: bool = False,
                         max_proxy_size: float = 20.0,
                         detect_slate: bool = False,
                         csv_path: Optional[Path] = None,
                         model: str = 'gemini-2.5-flash') -> Optional[VideoBackupEntry]:
    """Process a single video file and return backup entry."""
    _log(f"\nProcessing: {video_path.name}")
    _log(f"  Using model: {model}")

    _log("  Reading original metadata...")
    original_metadata = get_video_metadata(str(video_path))

    use_gemma = model == 'gemma-3-27b-it'

    if use_gemma:
        _log("  Analyzing with Gemma (frames + audio transcription)...")
        if detect_slate:
            _log("  (Slate detection enabled)")
        try:
            new_metadata = analyze_video_with_frames(str(video_path), client, detect_slate)
        except Exception as e:
            _log(f"  Error analyzing video: {e}")
            return None
    else:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            proxy_path = tmp.name

        try:
            _log("  Creating 360p proxy...")
            if not create_proxy(str(video_path), proxy_path, max_proxy_size):
                _log("  Error: Failed to create proxy")
                return None

            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            _log(f"  Proxy size: {proxy_size:.1f}MB")

            _log("  Analyzing with Gemini (full video)...")
            if detect_slate:
                _log("  (Slate detection enabled)")
            try:
                new_metadata = analyze_video(proxy_path, client, detect_slate)
            except Exception as e:
                _log(f"  Error analyzing video: {e}")
                return None
        finally:
            if os.path.exists(proxy_path):
                os.remove(proxy_path)

    _log(f"  Suggested filename: {new_metadata.filename}")
    _log(f"  Description: {new_metadata.description}")
    _log(f"  Tags: {', '.join(new_metadata.tags)}")

    new_filename = f"{new_metadata.filename}{video_path.suffix.lower()}"
    new_path = output_dir / new_filename

    # Handle duplicates
    counter = 1
    while new_path.exists() and new_path != video_path:
        new_filename = f"{new_metadata.filename}_{counter}{video_path.suffix.lower()}"
        new_path = output_dir / new_filename
        counter += 1

    if dry_run:
        _log(f"  [DRY RUN] Would rename to: {new_filename}")
        entry = VideoBackupEntry(
            original_path=str(video_path),
            original_filename=video_path.name,
            new_path=str(new_path),
            new_filename=new_filename,
            original_metadata=original_metadata,
            new_metadata=new_metadata,
            processed_at=datetime.now().isoformat()
        )
        if csv_path:
            append_to_davinci_csv(entry, csv_path, dry_run)
        return entry

    _log(f"  Writing metadata...")
    with tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False) as tmp_out:
        temp_output = tmp_out.name

    try:
        if write_metadata(str(video_path), temp_output,
                         new_metadata.filename, new_metadata.description, new_metadata.tags):
            shutil.move(temp_output, str(new_path))

            if video_path != new_path and video_path.exists():
                video_path.unlink()

            _log(f"  Renamed to: {new_filename}")

            entry = VideoBackupEntry(
                original_path=str(video_path),
                original_filename=video_path.name,
                new_path=str(new_path),
                new_filename=new_filename,
                original_metadata=original_metadata,
                new_metadata=new_metadata,
                processed_at=datetime.now().isoformat()
            )
            if csv_path:
                append_to_davinci_csv(entry, csv_path, dry_run)
            return entry
        else:
            _log("  Error: Failed to write metadata")
            return None
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)


def process_videos(video_files: List[Path], output_dir: Path,
                   client: genai.Client, dry_run: bool = False,
                   parallel: bool = False, max_workers: int = 4,
                   max_proxy_size: float = 20.0,
                   detect_slate: bool = False,
                   csv_path: Optional[Path] = None,
                   model: str = 'gemini-2.5-flash') -> List[VideoBackupEntry]:
    """Process multiple videos, optionally in parallel."""
    entries = []

    if parallel and len(video_files) > 1:
        _log(f"\nProcessing {len(video_files)} videos in parallel (max {max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_video, video_path, output_dir,
                    client, dry_run, max_proxy_size, detect_slate, csv_path, model
                ): video_path
                for video_path in video_files
            }

            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    entry = future.result()
                    if entry:
                        entries.append(entry)
                except Exception as e:
                    _log(f"Error processing {video_path.name}: {e}")
    else:
        for video_path in video_files:
            entry = process_single_video(video_path, output_dir, client,
                                         dry_run, max_proxy_size, detect_slate, csv_path, model)
            if entry:
                entries.append(entry)

    return entries


# ============ Public API (used by GUI and CLI) ============

def get_video_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Get all video files in directory."""
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f'*{ext}'))
        video_files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(set(video_files))


DEFAULT_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']


def run_processing(
    directory: str,
    *,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    extensions: Optional[List[str]] = None,
    parallel: bool = False,
    workers: int = 4,
    max_proxy_size: float = 20.0,
    detect_slate: bool = False,
    model: str = 'gemma-3-27b-it',
    on_progress: Optional[Callable] = None,
) -> dict:
    """Process videos in a directory.

    Args:
        directory: Path to directory containing videos.
        api_key: Google API key. If None, reads GOOGLE_API_KEY from env/.env.
        dry_run: Preview changes without writing anything.
        output_dir: Where to write renamed files. Defaults to input directory.
        extensions: Video extensions to scan. Defaults to DEFAULT_EXTENSIONS.
        parallel: Process videos concurrently.
        workers: Thread count when parallel=True.
        max_proxy_size: Proxy size cap in MB (Gemini model only).
        detect_slate: Extract scene/shot/take from clapperboard.
        model: 'gemini-2.5-flash' or 'gemma-3-27b-it'.
        on_progress: Callable(str) that receives log messages. Defaults to print.

    Returns:
        dict with keys: processed, total, backup_path, csv_path, entries.

    Raises:
        ValueError: If directory is invalid.
    """
    global _log
    _log = on_progress or print

    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key

    input_dir = Path(directory)
    if not input_dir.is_dir():
        raise ValueError(f"'{directory}' is not a valid directory")

    out_dir = Path(output_dir) if output_dir else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = extensions or DEFAULT_EXTENSIONS
    video_files = get_video_files(input_dir, exts)

    if not video_files:
        _log(f"No video files found in {input_dir}")
        return {'processed': 0, 'total': 0, 'backup_path': None, 'csv_path': None, 'entries': []}

    _log(f"Found {len(video_files)} video file(s)")
    for vf in video_files:
        _log(f"  - {vf.name}")

    _log("\nInitializing Gemini AI client...")
    client = genai.Client()

    backup = BackupLog(
        created_at=datetime.now().isoformat(),
        source_directory=str(input_dir),
        detect_slate=detect_slate,
    )

    csv_path = create_davinci_csv_filename(out_dir)

    entries = process_videos(
        video_files, out_dir, client,
        dry_run=dry_run,
        parallel=parallel,
        max_workers=workers,
        max_proxy_size=max_proxy_size,
        detect_slate=detect_slate,
        csv_path=csv_path,
        model=model,
    )

    backup.entries = entries

    backup_path_result = None
    csv_path_result = None

    if entries:
        backup_path_obj = create_backup_filename(input_dir)
        save_backup(backup, backup_path_obj)
        _log(f"\nBackup saved to: {backup_path_obj.name}")
        _log(f"To restore: python video_rename.py --restore {backup_path_obj}")
        backup_path_result = str(backup_path_obj)

        if csv_path.exists():
            _log(f"DaVinci CSV saved to: {csv_path.name}")
            csv_path_result = str(csv_path)

    _log(f"\nProcessed {len(entries)}/{len(video_files)} video(s) successfully")

    if dry_run:
        _log("\n[DRY RUN] No changes were made. Run without --dry-run to apply changes.")

    return {
        'processed': len(entries),
        'total': len(video_files),
        'backup_path': backup_path_result,
        'csv_path': csv_path_result,
        'entries': entries,
    }


def run_restore(
    backup_path: str,
    *,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> None:
    """Restore videos from a backup JSON file.

    Args:
        backup_path: Path to the backup JSON file.
        dry_run: Preview restore without writing anything.
        on_progress: Callable(str) that receives log messages. Defaults to print.

    Raises:
        FileNotFoundError: If backup_path does not exist.
    """
    global _log
    _log = on_progress or print

    path = Path(backup_path)
    if not path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    restore_from_backup(path, dry_run)


# ============ CLI ============

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI-powered video renaming and tagging tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s /path/to/videos                    # Process all videos in directory
  %(prog)s /path/to/videos --dry-run          # Preview changes without applying
  %(prog)s /path/to/videos --parallel         # Process videos in parallel
  %(prog)s /path/to/videos --detect-slate     # Detect slate/clapperboard info
  %(prog)s --restore backup.json              # Restore from backup file

Output:
  - JSON backup file for restore capability
  - DaVinci Resolve compatible CSV with metadata
        '''
    )

    parser.add_argument(
        'directory',
        nargs='?',
        type=str,
        help='Directory containing video files to process'
    )

    parser.add_argument(
        '--restore', '-r',
        type=str,
        metavar='BACKUP_FILE',
        help='Restore videos from a backup JSON file'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for renamed videos (default: same as input)'
    )

    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=DEFAULT_EXTENSIONS,
        help='Video file extensions to process (default: .mp4 .mov .avi .mkv .webm .m4v)'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel processing of videos'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, only used with --parallel)'
    )

    parser.add_argument(
        '--max-proxy-size', '-s',
        type=float,
        default=20.0,
        help='Maximum proxy file size in MB (default: 20)'
    )

    parser.add_argument(
        '--detect-slate',
        action='store_true',
        help='Enable slate/clapperboard detection for Scene, Shot, Take fields'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['gemini-2.5-flash', 'gemma-3-27b-it'],
        default='gemma-3-27b-it',
        help='AI model to use: gemini-2.5-flash (full video, only 20 requests per day) or gemma-3-27b-it (32 frames + audio transcription, 14.4k requests per day). Default: gemma-3-27b-it'
    )

    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_arguments()

    if args.restore:
        try:
            run_restore(args.restore, dry_run=args.dry_run)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        return 0

    if not args.directory:
        print("Error: Please provide a directory or use --restore")
        return 1

    try:
        run_processing(
            args.directory,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            extensions=args.extensions,
            parallel=args.parallel,
            workers=args.workers,
            max_proxy_size=args.max_proxy_size,
            detect_slate=args.detect_slate,
            model=args.model,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
