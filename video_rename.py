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
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


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
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def get_video_metadata(video_path: str) -> OriginalMetadata:
    """Extract existing metadata from video using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
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
        print(f"    Warning: Could not get duration, using default bitrate")
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
        'ffmpeg', '-y', '-i', input_path,
        '-vf', 'scale=-2:360',
        '-c:v', 'libx264', '-preset', 'fast',
        '-b:v', f'{target_bitrate}k',
        '-c:a', 'aac', '-b:a', f'{audio_bitrate}k',
        '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    FFmpeg error: {result.stderr[:500]}")
        return False
    
    # Verify file size
    actual_size = os.path.getsize(output_path) / (1024 * 1024)
    if actual_size > max_size_mb:
        print(f"    Warning: Proxy is {actual_size:.1f}MB (target was {max_size_mb}MB)")
    
    return True


def write_metadata(input_path: str, output_path: str,
                   title: str, description: str, tags: List[str]) -> bool:
    """Write metadata to video file using ffmpeg (stream copy, no re-encode)."""
    keywords = ', '.join(tags)
    
    result = subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-c', 'copy',
        '-movflags', 'use_metadata_tags',
        '-metadata', f'title={title}',
        '-metadata', f'description={description}',
        '-metadata', f'comment={description}',
        '-metadata', f'keywords={keywords}',
        output_path
    ], capture_output=True, text=True)
    
    return result.returncode == 0


def restore_metadata(input_path: str, output_path: str,
                     metadata: OriginalMetadata) -> bool:
    """Restore original metadata to video file."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
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
        print(f'  Waiting for video to be processed. {uploaded_file.name}:{uploaded_file.state}')
        time.sleep(5)
        print()
        uploaded_file = client.files.get(name=uploaded_file.name) 
    
    # Base prompt for video analysis
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
    
    # Add slate detection instructions if enabled
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

    # Retry logic for 503 UNAVAILABLE errors
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
            
            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
            
            return VideoMetadata.model_validate_json(response.text)
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a 503 UNAVAILABLE error
            if '503' in error_str and 'UNAVAILABLE' in error_str:
                if attempt < max_retries - 1:
                    print(f"  Model overloaded, waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"  Max retries reached. Model still overloaded.")
                    # Clean up uploaded file before raising
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception:
                        pass
                    raise
            else:
                # For other errors, clean up and raise immediately
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
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
    
    print(f"Restoring from backup: {backup_path.name}")
    print(f"Backup created at: {backup.created_at}")
    print(f"Entries to restore: {len(backup.entries)}")
    print()
    
    for entry in backup.entries:
        new_path = Path(entry.new_path)
        original_path = Path(entry.original_path)
        
        print(f"Restoring: {entry.new_filename} -> {entry.original_filename}")
        
        if not new_path.exists():
            print(f"  Warning: {new_path} not found, skipping")
            continue
        
        if dry_run:
            print(f"  [DRY RUN] Would restore metadata and rename")
            continue
        
        # Create temp file with restored metadata
        with tempfile.NamedTemporaryFile(suffix=new_path.suffix, delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Restore original metadata
            if restore_metadata(str(new_path), temp_path, entry.original_metadata):
                # Move temp file to original path
                shutil.move(temp_path, str(original_path))
                # Remove the renamed file if it's different from original
                if new_path != original_path and new_path.exists():
                    new_path.unlink()
                print(f"  Restored successfully")
            else:
                print(f"  Error: Failed to restore metadata")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"  Error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    print("\nRestore complete!")


# ============ DaVinci Resolve CSV Export ============

# CSV column order matching DaVinci Resolve metadata format
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
    """Append a single entry to DaVinci Resolve CSV, creating file if needed.
    
    Args:
        entry: Single VideoBackupEntry to append
        csv_path: Path to CSV file
        dry_run: If True, use backup timestamps instead of actual file stats
        
    Returns:
        True if append was successful, False otherwise
    """
    try:
        # Check if file exists to determine if we need to write header
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DAVINCI_CSV_COLUMNS)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            new_path = Path(entry.new_path)
            
            # Format comments as "Description: <desc>\nTags: <tags>"
            tags_str = ', '.join(entry.new_metadata.tags)
            comments = f"Description: {entry.new_metadata.description}\nTags: {tags_str}"
            
            # Get date modified from file or use processed_at timestamp
            if not dry_run and new_path.exists():
                date_modified = format_date_modified(new_path)
            else:
                # Use processed_at timestamp for dry run
                try:
                    dt = datetime.fromisoformat(entry.processed_at)
                    date_modified = dt.strftime("%a %b %d %H:%M:%S %Y")
                except ValueError:
                    date_modified = ""
            
            # Build row with all columns (technical fields empty, slate from metadata)
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
        print(f"  Warning: Failed to update CSV: {e}")
        return False


def export_davinci_csv(backup: BackupLog, csv_path: Path, dry_run: bool = False) -> bool:
    """Export backup entries to DaVinci Resolve compatible CSV.
    
    Args:
        backup: BackupLog containing processed video entries
        csv_path: Path to write CSV file
        dry_run: If True, use backup timestamps instead of actual file stats
        
    Returns:
        True if export was successful, False otherwise
    """
    if not backup.entries:
        print("No entries to export to CSV")
        return False
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DAVINCI_CSV_COLUMNS)
            writer.writeheader()
            
            for entry in backup.entries:
                new_path = Path(entry.new_path)
                clip_directory = str(new_path.parent)
                
                # Format comments as "Description: <desc>\nTags: <tags>"
                tags_str = ', '.join(entry.new_metadata.tags)
                comments = f"Description: {entry.new_metadata.description}\nTags: {tags_str}"
                
                # Get date modified from file or use processed_at timestamp
                if not dry_run and new_path.exists():
                    date_modified = format_date_modified(new_path)
                else:
                    # Use processed_at timestamp for dry run
                    try:
                        dt = datetime.fromisoformat(entry.processed_at)
                        date_modified = dt.strftime("%a %b %d %H:%M:%S %Y")
                    except ValueError:
                        date_modified = ""
                
                # Build row with all columns (technical fields empty, slate from metadata)
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
        print(f"Error exporting CSV: {e}")
        return False


# ============ Video Processing ============

def process_single_video(video_path: Path, output_dir: Path, 
                         client: genai.Client, dry_run: bool = False,
                         max_proxy_size: float = 20.0,
                         detect_slate: bool = False,
                         csv_path: Optional[Path] = None) -> Optional[VideoBackupEntry]:
    """Process a single video file and return backup entry."""
    print(f"\nProcessing: {video_path.name}")
    
    # Get original metadata
    print("  Reading original metadata...")
    original_metadata = get_video_metadata(str(video_path))
    
    # Create proxy
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        proxy_path = tmp.name
    
    try:
        print("  Creating 360p proxy...")
        if not create_proxy(str(video_path), proxy_path, max_proxy_size):
            print("  Error: Failed to create proxy")
            return None
        
        proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
        print(f"  Proxy size: {proxy_size:.1f}MB")
        
        # Analyze with AI
        print("  Analyzing with AI...")
        if detect_slate:
            print("  (Slate detection enabled)")
        try:
            new_metadata = analyze_video(proxy_path, client, detect_slate)
        except Exception as e:
            print(f"  Error analyzing video: {e}")
            return None
        
        print(f"  Suggested filename: {new_metadata.filename}")
        print(f"  Description: {new_metadata.description}")
        print(f"  Tags: {', '.join(new_metadata.tags)}")
        
        # Create new filename
        new_filename = f"{new_metadata.filename}{video_path.suffix.lower()}"
        new_path = output_dir / new_filename
        
        # Handle duplicates
        counter = 1
        while new_path.exists() and new_path != video_path:
            new_filename = f"{new_metadata.filename}_{counter}{video_path.suffix.lower()}"
            new_path = output_dir / new_filename
            counter += 1
        
        if dry_run:
            print(f"  [DRY RUN] Would rename to: {new_filename}")
            entry = VideoBackupEntry(
                original_path=str(video_path),
                original_filename=video_path.name,
                new_path=str(new_path),
                new_filename=new_filename,
                original_metadata=original_metadata,
                new_metadata=new_metadata,
                processed_at=datetime.now().isoformat()
            )
            # Update CSV immediately after processing
            if csv_path:
                append_to_davinci_csv(entry, csv_path, dry_run)
            return entry
        
        # Write metadata to temp file and move
        print(f"  Writing metadata...")
        with tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False) as tmp_out:
            temp_output = tmp_out.name
        
        if write_metadata(str(video_path), temp_output,
                         new_metadata.filename, new_metadata.description, new_metadata.tags):
            # Move to final destination
            shutil.move(temp_output, str(new_path))
            
            # Remove original if different from new path
            if video_path != new_path and video_path.exists():
                video_path.unlink()
            
            print(f"  Renamed to: {new_filename}")
            
            entry = VideoBackupEntry(
                original_path=str(video_path),
                original_filename=video_path.name,
                new_path=str(new_path),
                new_filename=new_filename,
                original_metadata=original_metadata,
                new_metadata=new_metadata,
                processed_at=datetime.now().isoformat()
            )
            # Update CSV immediately after processing
            if csv_path:
                append_to_davinci_csv(entry, csv_path, dry_run)
            return entry
        else:
            print("  Error: Failed to write metadata")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return None
            
    finally:
        # Cleanup proxy
        if os.path.exists(proxy_path):
            os.remove(proxy_path)


def process_videos(video_files: List[Path], output_dir: Path,
                   client: genai.Client, dry_run: bool = False,
                   parallel: bool = False, max_workers: int = 4,
                   max_proxy_size: float = 20.0,
                   detect_slate: bool = False,
                   csv_path: Optional[Path] = None) -> List[VideoBackupEntry]:
    """Process multiple videos, optionally in parallel."""
    entries = []
    
    if parallel and len(video_files) > 1:
        print(f"\nProcessing {len(video_files)} videos in parallel (max {max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_video, video_path, output_dir, 
                    client, dry_run, max_proxy_size, detect_slate, csv_path
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
                    print(f"Error processing {video_path.name}: {e}")
    else:
        for video_path in video_files:
            entry = process_single_video(video_path, output_dir, client, 
                                         dry_run, max_proxy_size, detect_slate, csv_path)
            if entry:
                entries.append(entry)
    
    return entries


# ============ CLI ============

def get_video_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Get all video files in directory."""
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f'*{ext}'))
        video_files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(set(video_files))


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
    
    # Main arguments
    parser.add_argument(
        'directory',
        nargs='?',
        type=str,
        help='Directory containing video files to process'
    )
    
    # Restore mode
    parser.add_argument(
        '--restore', '-r',
        type=str,
        metavar='BACKUP_FILE',
        help='Restore videos from a backup JSON file'
    )
    
    # Processing options
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
        default=['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'],
        help='Video file extensions to process (default: .mp4 .mov .avi .mkv .webm .m4v)'
    )
    
    # Parallel processing
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
    
    # Proxy settings
    parser.add_argument(
        '--max-proxy-size', '-s',
        type=float,
        default=20.0,
        help='Maximum proxy file size in MB (default: 20)'
    )
    
    # Slate detection
    parser.add_argument(
        '--detect-slate',
        action='store_true',
        help='Enable slate/clapperboard detection for Scene, Shot, Take fields'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Restore mode
    if args.restore:
        backup_path = Path(args.restore)
        if not backup_path.exists():
            print(f"Error: Backup file not found: {args.restore}")
            return 1
        restore_from_backup(backup_path, args.dry_run)
        return 0
    
    # Process mode requires directory
    if not args.directory:
        print("Error: Please provide a directory or use --restore")
        return 1
    
    input_dir = Path(args.directory)
    if not input_dir.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = get_video_files(input_dir, args.extensions)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return 0
    
    print(f"Found {len(video_files)} video file(s)")
    for vf in video_files:
        print(f"  - {vf.name}")
    
    # Initialize Gemini client
    print("\nInitializing Gemini AI client...")
    client = genai.Client()
    
    # Create backup log
    backup = BackupLog(
        created_at=datetime.now().isoformat(),
        source_directory=str(input_dir),
        detect_slate=args.detect_slate
    )
    
    # Create CSV path upfront
    csv_path = create_davinci_csv_filename(output_dir)
    
    # Process videos (CSV updated incrementally during processing)
    entries = process_videos(
        video_files, output_dir, client,
        dry_run=args.dry_run,
        parallel=args.parallel,
        max_workers=args.workers,
        max_proxy_size=args.max_proxy_size,
        detect_slate=args.detect_slate,
        csv_path=csv_path
    )
    
    backup.entries = entries
    
    # Save backup (even for dry run, so user can see what would happen)
    if entries:
        backup_path = create_backup_filename(input_dir)
        save_backup(backup, backup_path)
        print(f"\nBackup saved to: {backup_path.name}")
        print(f"To restore: python video_rename.py --restore {backup_path}")
        
        # CSV was already updated incrementally during processing
        if csv_path.exists():
            print(f"DaVinci CSV saved to: {csv_path.name}")
    
    print(f"\nProcessed {len(entries)}/{len(video_files)} video(s) successfully")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply changes.")
    
    return 0


if __name__ == '__main__':
    exit(main())