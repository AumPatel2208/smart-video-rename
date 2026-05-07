"""Shared fixtures and import-time mocks for all test modules."""

import sys
from unittest.mock import MagicMock

# google.genai has a broken transitive dependency (cryptography/cffi) in some
# environments.  Mock it out here — before any import of video_rename — so
# the test suite runs anywhere without a real Google API key.
sys.modules.setdefault('google.genai', MagicMock())

import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from video_rename import (  # noqa: E402
    BackupLog, VideoBackupEntry, VideoMetadata, OriginalMetadata,
)


@pytest.fixture
def sample_entry(tmp_path):
    return VideoBackupEntry(
        original_path=str(tmp_path / 'old.mp4'),
        original_filename='old.mp4',
        new_path=str(tmp_path / 'new_name.mp4'),
        new_filename='new_name.mp4',
        original_metadata=OriginalMetadata(title='Original Title', description='Old desc'),
        new_metadata=VideoMetadata(
            filename='new_name',
            description='A descriptive video about testing',
            tags=['test', 'video', 'demo'],
        ),
        processed_at='2024-06-01T12:00:00',
    )


@pytest.fixture
def sample_backup(tmp_path, sample_entry):
    return BackupLog(
        created_at='2024-06-01T00:00:00',
        source_directory=str(tmp_path),
        entries=[sample_entry],
    )
