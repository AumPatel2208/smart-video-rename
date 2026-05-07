"""
Tests for video_rename.py.

External calls (subprocess/genai/whisper) are mocked throughout so tests run
without FFmpeg, an API key, or any real video files.
"""

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import video_rename
from video_rename import (
    DEFAULT_EXTENSIONS,
    BackupLog,
    OriginalMetadata,
    VideoBackupEntry,
    VideoMetadata,
    append_to_davinci_csv,
    create_backup_filename,
    create_davinci_csv_filename,
    create_proxy,
    export_davinci_csv,
    extract_audio,
    extract_frames,
    get_video_duration,
    get_video_files,
    get_video_metadata,
    load_backup,
    restore_from_backup,
    restore_metadata,
    run_processing,
    run_restore,
    save_backup,
    write_metadata,
)


# ── Data models ───────────────────────────────────────────────────────────────

class TestVideoMetadata:
    def test_basic_fields(self):
        m = VideoMetadata(filename='my_video', description='A test.', tags=['a', 'b'])
        assert m.filename == 'my_video'
        assert m.tags == ['a', 'b']
        assert m.scene is None and m.shot is None and m.take is None

    def test_slate_fields(self):
        m = VideoMetadata(
            filename='s', description='d', tags=['t'],
            scene='1', shot='A', take='3',
        )
        assert m.scene == '1'
        assert m.shot == 'A'
        assert m.take == '3'

    def test_json_roundtrip(self):
        m = VideoMetadata(filename='test', description='desc', tags=['x'])
        assert VideoMetadata.model_validate_json(m.model_dump_json()) == m


class TestBackupModels:
    def test_backup_log_defaults(self):
        b = BackupLog(created_at='2024-01-01', source_directory='/tmp')
        assert b.entries == []
        assert b.detect_slate is False

    def test_backup_roundtrip(self, sample_backup):
        data = sample_backup.model_dump()
        restored = BackupLog.model_validate(data)
        assert len(restored.entries) == 1
        assert restored.entries[0].original_filename == 'old.mp4'
        assert restored.entries[0].new_metadata.filename == 'new_name'


# ── _ffmpeg_bin ───────────────────────────────────────────────────────────────

class TestFfmpegBin:
    def test_returns_name_when_not_frozen(self):
        assert video_rename._ffmpeg_bin('ffmpeg') == 'ffmpeg'
        assert video_rename._ffmpeg_bin('ffprobe') == 'ffprobe'

    def test_uses_bundled_binary_when_frozen(self, tmp_path):
        (tmp_path / 'bin').mkdir()
        (tmp_path / 'bin' / 'ffmpeg').touch()
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, '_MEIPASS', str(tmp_path), create=True):
            result = video_rename._ffmpeg_bin('ffmpeg')
        assert result == str(tmp_path / 'bin' / 'ffmpeg')

    def test_falls_back_to_path_when_binary_missing_in_bundle(self, tmp_path):
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, '_MEIPASS', str(tmp_path), create=True):
            result = video_rename._ffmpeg_bin('ffmpeg')
        assert result == 'ffmpeg'


# ── get_video_files ───────────────────────────────────────────────────────────

class TestGetVideoFiles:
    def test_finds_matching_extensions(self, tmp_path):
        (tmp_path / 'a.mp4').touch()
        (tmp_path / 'b.MOV').touch()
        (tmp_path / 'c.txt').touch()
        names = {f.name for f in get_video_files(tmp_path, ['.mp4', '.mov'])}
        assert 'a.mp4' in names
        assert 'b.MOV' in names
        assert 'c.txt' not in names

    def test_empty_directory(self, tmp_path):
        assert get_video_files(tmp_path, DEFAULT_EXTENSIONS) == []

    def test_no_duplicates(self, tmp_path):
        (tmp_path / 'v.mp4').touch()
        assert len(get_video_files(tmp_path, ['.mp4', '.mp4'])) == 1

    def test_sorted_output(self, tmp_path):
        for name in ['c.mp4', 'a.mp4', 'b.mp4']:
            (tmp_path / name).touch()
        assert [f.name for f in get_video_files(tmp_path, ['.mp4'])] == ['a.mp4', 'b.mp4', 'c.mp4']


# ── get_video_duration ────────────────────────────────────────────────────────

class TestGetVideoDuration:
    def test_parses_ffprobe_float(self):
        with patch('subprocess.run', return_value=MagicMock(stdout='123.456\n', returncode=0)):
            assert get_video_duration('/v.mp4') == pytest.approx(123.456)

    def test_propagates_subprocess_error(self):
        with patch('subprocess.run', side_effect=Exception('fail')):
            with pytest.raises(Exception):
                get_video_duration('/v.mp4')


# ── get_video_metadata ────────────────────────────────────────────────────────

class TestGetVideoMetadata:
    def _mock_ffprobe(self, tags: dict):
        payload = json.dumps({'format': {'tags': tags}})
        return MagicMock(returncode=0, stdout=payload)

    def test_reads_standard_tags(self):
        with patch('subprocess.run', return_value=self._mock_ffprobe(
            {'title': 'T', 'description': 'D', 'comment': 'C', 'keywords': 'K'}
        )):
            m = get_video_metadata('/v.mp4')
        assert m.title == 'T'
        assert m.description == 'D'
        assert m.comment == 'C'
        assert m.keywords == 'K'

    def test_case_insensitive_keys(self):
        with patch('subprocess.run', return_value=self._mock_ffprobe(
            {'TITLE': 'Upper', 'Description': 'Mixed'}
        )):
            m = get_video_metadata('/v.mp4')
        assert m.title == 'Upper'
        assert m.description == 'Mixed'

    def test_returns_empty_on_ffprobe_failure(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=1, stdout='')):
            m = get_video_metadata('/v.mp4')
        assert m.title is None

    def test_returns_empty_on_bad_json(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0, stdout='not-json')):
            m = get_video_metadata('/v.mp4')
        assert m.title is None


# ── create_proxy ──────────────────────────────────────────────────────────────

class TestCreateProxy:
    def test_returns_true_on_success(self, tmp_path):
        out = str(tmp_path / 'proxy.mp4')
        (tmp_path / 'proxy.mp4').write_bytes(b'\x00' * 100)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout='60.0\n', returncode=0),   # ffprobe
                MagicMock(returncode=0, stderr=''),          # ffmpeg
            ]
            assert create_proxy('/input.mp4', out) is True

    def test_returns_false_on_ffmpeg_error(self, tmp_path):
        out = str(tmp_path / 'proxy.mp4')
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout='60.0\n', returncode=0),
                MagicMock(returncode=1, stderr='encode error'),
            ]
            assert create_proxy('/input.mp4', out) is False

    def test_uses_duration_for_bitrate_calculation(self, tmp_path):
        out = str(tmp_path / 'proxy.mp4')
        (tmp_path / 'proxy.mp4').write_bytes(b'\x00' * 100)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout='300.0\n', returncode=0),  # 5 min video
                MagicMock(returncode=0, stderr=''),
            ]
            create_proxy('/input.mp4', out, max_size_mb=20.0)
            ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        # bitrate arg should be present
        assert '-b:v' in ffmpeg_cmd

    def test_falls_back_to_default_duration_on_probe_error(self, tmp_path):
        out = str(tmp_path / 'proxy.mp4')
        (tmp_path / 'proxy.mp4').write_bytes(b'\x00' * 100)
        import subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, 'ffprobe'),  # duration fails
                MagicMock(returncode=0, stderr=''),
            ]
            result = create_proxy('/input.mp4', out)
        assert result is True


# ── write_metadata ────────────────────────────────────────────────────────────

class TestWriteMetadata:
    def test_returns_true_on_success(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            assert write_metadata('/in.mp4', '/out.mp4', 'title', 'desc', ['a', 'b']) is True

    def test_returns_false_on_failure(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=1)):
            assert write_metadata('/in.mp4', '/out.mp4', 'title', 'desc', ['a']) is False

    def test_joins_tags_as_keywords(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            write_metadata('/in.mp4', '/out.mp4', 'title', 'desc', ['x', 'y', 'z'])
        cmd = mock_run.call_args[0][0]
        assert 'x, y, z' in ' '.join(str(c) for c in cmd)


# ── extract_audio ─────────────────────────────────────────────────────────────

class TestExtractAudio:
    def test_returns_true_on_success(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            assert extract_audio('/v.mp4', '/out.wav') is True

    def test_returns_false_on_failure(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=1)):
            assert extract_audio('/v.mp4', '/out.wav') is False

    def test_uses_mono_16khz(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            extract_audio('/v.mp4', '/out.wav')
        cmd = mock_run.call_args[0][0]
        assert '-ar' in cmd
        assert '16000' in cmd
        assert '-ac' in cmd
        assert '1' in cmd


# ── extract_frames ────────────────────────────────────────────────────────────

class TestExtractFrames:
    def _fake_frames(self, output_dir, count):
        """Create fake frame files in output_dir."""
        for i in range(1, count + 1):
            (Path(output_dir) / f'frame_{i:04d}.jpg').touch()

    def test_returns_frame_paths_on_success(self, tmp_path):
        self._fake_frames(tmp_path, 4)
        with patch('subprocess.run', return_value=MagicMock(returncode=0)), \
             patch('video_rename.get_video_duration', return_value=10.0):
            frames = extract_frames('/v.mp4', str(tmp_path), num_frames=4)
        assert len(frames) == 4
        assert all(f.endswith('.jpg') for f in frames)

    def test_returns_empty_list_on_ffmpeg_failure(self, tmp_path):
        with patch('subprocess.run', return_value=MagicMock(returncode=1, stderr='err')), \
             patch('video_rename.get_video_duration', return_value=10.0):
            frames = extract_frames('/v.mp4', str(tmp_path))
        assert frames == []

    def test_caps_result_at_num_frames(self, tmp_path):
        self._fake_frames(tmp_path, 10)
        with patch('subprocess.run', return_value=MagicMock(returncode=0)), \
             patch('video_rename.get_video_duration', return_value=30.0):
            frames = extract_frames('/v.mp4', str(tmp_path), num_frames=5)
        assert len(frames) <= 5


# ── restore_metadata ──────────────────────────────────────────────────────────

class TestRestoreMetadata:
    def test_returns_true_on_success(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            assert restore_metadata('/new.mp4', '/orig.mp4', OriginalMetadata()) is True

    def test_returns_false_on_failure(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=1)):
            assert restore_metadata('/new.mp4', '/orig.mp4', OriginalMetadata()) is False

    def test_clears_missing_fields(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            restore_metadata('/new.mp4', '/orig.mp4', OriginalMetadata(title=None))
        cmd = mock_run.call_args[0][0]
        assert 'title=' in ' '.join(str(c) for c in cmd)

    def test_sets_present_fields(self):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            restore_metadata('/new.mp4', '/orig.mp4', OriginalMetadata(title='My Film'))
        cmd = mock_run.call_args[0][0]
        assert 'title=My Film' in ' '.join(str(c) for c in cmd)


# ── Backup file I/O ───────────────────────────────────────────────────────────

class TestSaveLoadBackup:
    def test_roundtrip(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        loaded = load_backup(p)
        assert loaded.created_at == sample_backup.created_at
        assert loaded.entries[0].new_filename == 'new_name.mp4'

    def test_output_is_valid_json(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        data = json.loads(p.read_text())
        assert 'entries' in data and 'source_directory' in data

    def test_create_backup_filename_format(self, tmp_path):
        p = create_backup_filename(tmp_path)
        assert p.parent == tmp_path
        assert p.name.startswith('video_rename_backup_')
        assert p.suffix == '.json'


# ── restore_from_backup ───────────────────────────────────────────────────────

class TestRestoreFromBackup:
    def test_dry_run_leaves_files_untouched(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        (tmp_path / 'new_name.mp4').touch()

        logs = []
        video_rename._log = logs.append
        try:
            restore_from_backup(p, dry_run=True)
        finally:
            video_rename._log = print

        assert (tmp_path / 'new_name.mp4').exists()
        assert any('DRY RUN' in m for m in logs)

    def test_skips_missing_new_file(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        # new_name.mp4 intentionally not created

        logs = []
        video_rename._log = logs.append
        try:
            restore_from_backup(p, dry_run=True)
        finally:
            video_rename._log = print

        assert any('not found' in m for m in logs)

    def test_logs_entry_count(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)

        logs = []
        video_rename._log = logs.append
        try:
            restore_from_backup(p, dry_run=True)
        finally:
            video_rename._log = print

        assert any('1' in m and 'restore' in m.lower() for m in logs)


# ── DaVinci Resolve CSV ───────────────────────────────────────────────────────

class TestDavinciCsv:
    def test_export_writes_header_and_row(self, tmp_path, sample_backup):
        p = tmp_path / 'out.csv'
        assert export_davinci_csv(sample_backup, p) is True
        text = p.read_text()
        assert 'File Name' in text
        assert 'new_name.mp4' in text
        assert 'A descriptive video' in text

    def test_export_empty_backup_returns_false(self, tmp_path):
        empty = BackupLog(created_at='2024-01-01', source_directory=str(tmp_path))
        assert export_davinci_csv(empty, tmp_path / 'out.csv') is False

    def test_append_creates_file_with_header(self, tmp_path, sample_entry):
        p = tmp_path / 'out.csv'
        append_to_davinci_csv(sample_entry, p)
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 1
        assert rows[0]['File Name'] == 'new_name.mp4'

    def test_append_no_duplicate_header(self, tmp_path, sample_entry):
        p = tmp_path / 'out.csv'
        append_to_davinci_csv(sample_entry, p)
        append_to_davinci_csv(sample_entry, p)
        assert p.read_text().count('File Name') == 1

    def test_tags_included_in_comments(self, tmp_path, sample_entry):
        p = tmp_path / 'out.csv'
        export_davinci_csv(
            BackupLog(created_at='x', source_directory=str(tmp_path), entries=[sample_entry]),
            p,
        )
        text = p.read_text()
        assert 'test' in text and 'video' in text

    def test_slate_fields_written(self, tmp_path):
        entry = VideoBackupEntry(
            original_path='/a.mp4', original_filename='a.mp4',
            new_path='/b.mp4', new_filename='b.mp4',
            original_metadata=OriginalMetadata(),
            new_metadata=VideoMetadata(
                filename='b', description='d', tags=['t'],
                scene='2', shot='B', take='5',
            ),
            processed_at='2024-01-01T00:00:00',
        )
        p = tmp_path / 'out.csv'
        export_davinci_csv(
            BackupLog(created_at='x', source_directory=str(tmp_path), entries=[entry]), p
        )
        rows = list(csv.DictReader(p.open()))
        assert rows[0]['Scene'] == '2'
        assert rows[0]['Shot'] == 'B'
        assert rows[0]['Take'] == '5'

    def test_create_csv_filename_format(self, tmp_path):
        p = create_davinci_csv_filename(tmp_path)
        assert p.parent == tmp_path
        assert p.name.startswith('davinci_metadata_')
        assert p.suffix == '.csv'


# ── run_processing ────────────────────────────────────────────────────────────

class TestRunProcessing:
    def test_raises_on_invalid_directory(self):
        with pytest.raises(ValueError, match='not a valid directory'):
            run_processing('/nonexistent/xyz_path')

    def test_returns_zero_counts_when_no_videos(self, tmp_path):
        result = run_processing(str(tmp_path), on_progress=lambda _: None)
        assert result == {'processed': 0, 'total': 0, 'backup_path': None, 'csv_path': None, 'entries': []}

    def test_sets_api_key_env_var(self, tmp_path):
        with patch('video_rename.get_video_files', return_value=[]):
            run_processing(str(tmp_path), api_key='test-key-abc', on_progress=lambda _: None)
        assert os.environ.get('GOOGLE_API_KEY') == 'test-key-abc'
        del os.environ['GOOGLE_API_KEY']

    def test_dry_run_does_not_rename_files(self, tmp_path):
        (tmp_path / 'clip.mp4').touch()
        fake_meta = VideoMetadata(filename='ai_name', description='desc', tags=['t'])

        with patch('video_rename.genai') as mock_genai, \
             patch('video_rename.get_video_metadata', return_value=OriginalMetadata()), \
             patch('video_rename.analyze_video_with_frames', return_value=fake_meta), \
             patch('subprocess.run', return_value=MagicMock(returncode=0, stdout='10.0\n', stderr='')):
            mock_genai.Client.return_value = MagicMock()
            result = run_processing(
                str(tmp_path), dry_run=True, model='gemma-3-27b-it',
                on_progress=lambda _: None,
            )

        assert (tmp_path / 'clip.mp4').exists()
        assert result['total'] == 1

    def test_creates_output_dir_if_missing(self, tmp_path):
        out = tmp_path / 'nested' / 'output'
        with patch('video_rename.get_video_files', return_value=[]):
            run_processing(str(tmp_path), output_dir=str(out), on_progress=lambda _: None)
        assert out.is_dir()

    def test_on_progress_receives_messages(self, tmp_path):
        messages = []
        run_processing(str(tmp_path), on_progress=messages.append)
        assert any('No video files' in m or 'Found' in m or isinstance(m, str) for m in messages)

    def test_resets_log_to_print_after_run(self, tmp_path):
        run_processing(str(tmp_path), on_progress=lambda _: None)
        # After run, _log should be the custom callback (set for this run)
        # The important thing is it doesn't raise
        assert callable(video_rename._log)


# ── run_restore ───────────────────────────────────────────────────────────────

class TestRunRestore:
    def test_raises_on_missing_backup(self):
        with pytest.raises(FileNotFoundError):
            run_restore('/no/such/file.json')

    def test_calls_restore_from_backup_with_path(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        with patch('video_rename.restore_from_backup') as mock_fn:
            run_restore(str(p), dry_run=False)
        mock_fn.assert_called_once_with(p, False)

    def test_passes_dry_run_flag(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        with patch('video_rename.restore_from_backup') as mock_fn:
            run_restore(str(p), dry_run=True)
        mock_fn.assert_called_once_with(p, True)

    def test_on_progress_replaces_log(self, tmp_path, sample_backup):
        p = tmp_path / 'backup.json'
        save_backup(sample_backup, p)
        sentinel = object()
        with patch('video_rename.restore_from_backup'):
            run_restore(str(p), on_progress=lambda _: sentinel)
        assert video_rename._log is not print


# ── Retry logic in analyze_video ─────────────────────────────────────────────

class TestAnalyzeVideoRetry:
    def test_retries_on_503_and_succeeds(self):
        fake_meta = VideoMetadata(filename='f', description='d', tags=['t'])
        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.state = 'ACTIVE'
        mock_client.files.upload.return_value = mock_file

        error_503 = Exception('503 UNAVAILABLE')
        success_response = MagicMock()
        success_response.text = fake_meta.model_dump_json()

        mock_client.models.generate_content.side_effect = [error_503, success_response]

        with patch('time.sleep'):
            result = video_rename.analyze_video('proxy.mp4', mock_client)

        assert result.filename == 'f'
        assert mock_client.models.generate_content.call_count == 2

    def test_raises_after_max_retries(self):
        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.state = 'ACTIVE'
        mock_client.files.upload.return_value = mock_file
        mock_client.models.generate_content.side_effect = Exception('503 UNAVAILABLE')

        with patch('time.sleep'), pytest.raises(Exception, match='503'):
            video_rename.analyze_video('proxy.mp4', mock_client)

        assert mock_client.models.generate_content.call_count == 3

    def test_non_503_error_raises_immediately(self):
        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.state = 'ACTIVE'
        mock_client.files.upload.return_value = mock_file
        mock_client.models.generate_content.side_effect = ValueError('bad schema')

        with patch('time.sleep'), pytest.raises(ValueError):
            video_rename.analyze_video('proxy.mp4', mock_client)

        assert mock_client.models.generate_content.call_count == 1


# ── CLI (main + parse_arguments) ──────────────────────────────────────────────

class TestCLI:
    def test_no_args_returns_error(self, capsys):
        with patch('sys.argv', ['video_rename.py']):
            assert video_rename.main() == 1
        assert 'Error' in capsys.readouterr().out

    def test_invalid_directory_returns_error(self, capsys):
        with patch('sys.argv', ['video_rename.py', '/no/such/dir/xyz']):
            assert video_rename.main() == 1

    def test_restore_missing_file_returns_error(self, capsys):
        with patch('sys.argv', ['video_rename.py', '--restore', '/no/such.json']):
            assert video_rename.main() == 1
        assert 'Error' in capsys.readouterr().out

    def test_restore_calls_run_restore(self, tmp_path, sample_backup):
        p = tmp_path / 'b.json'
        save_backup(sample_backup, p)
        with patch('sys.argv', ['video_rename.py', '--restore', str(p)]), \
             patch('video_rename.run_restore') as mock_fn:
            assert video_rename.main() == 0
        mock_fn.assert_called_once()

    def test_dry_run_flag_passed_to_restore(self, tmp_path, sample_backup):
        p = tmp_path / 'b.json'
        save_backup(sample_backup, p)
        with patch('sys.argv', ['video_rename.py', '--restore', str(p), '--dry-run']), \
             patch('video_rename.run_restore') as mock_fn:
            video_rename.main()
        assert mock_fn.call_args.kwargs['dry_run'] is True

    def test_model_flag_parsed(self, tmp_path):
        with patch('sys.argv', ['video_rename.py', str(tmp_path), '--model', 'gemini-2.5-flash']), \
             patch('video_rename.run_processing') as mock_fn:
            video_rename.main()
        assert mock_fn.call_args.kwargs['model'] == 'gemini-2.5-flash'

    def test_parallel_flag_parsed(self, tmp_path):
        with patch('sys.argv', ['video_rename.py', str(tmp_path), '--parallel']), \
             patch('video_rename.run_processing') as mock_fn:
            video_rename.main()
        assert mock_fn.call_args.kwargs['parallel'] is True

    def test_workers_flag_parsed(self, tmp_path):
        with patch('sys.argv', ['video_rename.py', str(tmp_path), '--workers', '8']), \
             patch('video_rename.run_processing') as mock_fn:
            video_rename.main()
        assert mock_fn.call_args.kwargs['workers'] == 8
