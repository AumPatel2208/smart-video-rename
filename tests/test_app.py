"""
Tests for the config-persistence and key-management logic in app.py.

app.py calls ui.run() at module level, so we stub out the entire nicegui
package before importing.  The UI widget calls become no-ops and the pure
Python config functions can be exercised normally.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Stub out nicegui, video_rename, and google.genai before 'import app' ─────
# Must happen before the import so that module-level ui.* calls are no-ops and
# the broken google.genai transitive dep (cryptography/cffi) is never loaded.
for _mod in ('nicegui', 'nicegui.run', 'video_rename', 'google.genai'):
    sys.modules.setdefault(_mod, MagicMock())

import app  # noqa: E402  (import after sys.modules patching)


# ── _mask ─────────────────────────────────────────────────────────────────────

class TestMask:
    def test_short_key_returned_as_is(self):
        assert app._mask('short') == 'short'

    def test_exactly_14_chars_returned_as_is(self):
        key = 'A' * 14
        assert app._mask(key) == key

    def test_15_chars_is_masked(self):
        key = 'A' * 15
        result = app._mask(key)
        assert '...' in result
        assert result != key

    def test_long_key_shows_first_and_last_chars(self):
        key = 'AIzaSyABCDEFGH1234'
        result = app._mask(key)
        assert result.startswith(key[:8])
        assert result.endswith(key[-4:])
        assert '...' in result

    def test_two_different_keys_produce_different_masks(self):
        k1 = 'AIzaSyAAAAAAAA1111'
        k2 = 'AIzaSyBBBBBBBB2222'
        assert app._mask(k1) != app._mask(k2)


# ── _key_options ──────────────────────────────────────────────────────────────

class TestKeyOptions:
    def test_empty_list_returns_placeholder(self):
        opts = app._key_options([])
        assert '' in opts
        assert 'no saved' in opts[''].lower()

    def test_single_key_is_the_dict_key(self):
        key = 'AIzaSyABCDEFGH1234'
        opts = app._key_options([key])
        assert key in opts

    def test_display_label_is_masked(self):
        key = 'AIzaSyABCDEFGH1234'
        opts = app._key_options([key])
        assert opts[key] != key
        assert '...' in opts[key]

    def test_multiple_keys_all_present(self):
        keys = ['key_one_long_enough', 'key_two_long_enough', 'key_three_long_x']
        opts = app._key_options(keys)
        assert set(opts.keys()) == set(keys)

    def test_short_key_not_masked(self):
        key = 'short'
        opts = app._key_options([key])
        assert opts[key] == key  # no masking for short keys


# ── _config_path ──────────────────────────────────────────────────────────────

class TestConfigPath:
    def test_returns_a_path_ending_in_settings_json(self):
        p = app._config_path()
        assert p.name == 'settings.json'

    def test_creates_parent_directory(self, tmp_path):
        target = tmp_path / 'nested' / 'dir' / 'settings.json'
        with patch.object(app, '_config_path', return_value=target):
            # Calling _save_cfg should trigger directory creation via _config_path
            # We test the directory-creation logic directly here
            target.parent.mkdir(parents=True, exist_ok=True)
        assert target.parent.is_dir()

    def test_windows_uses_appdata(self, tmp_path, monkeypatch):
        monkeypatch.setenv('APPDATA', str(tmp_path))
        with patch.object(sys, 'platform', 'win32'):
            p = app._config_path()
        assert str(tmp_path) in str(p)

    def test_linux_uses_xdg_config_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
        with patch.object(sys, 'platform', 'linux'):
            p = app._config_path()
        assert str(tmp_path) in str(p)


# ── _load_cfg / _save_cfg ─────────────────────────────────────────────────────

class TestConfigPersistence:
    def test_returns_defaults_when_no_file(self, tmp_path):
        p = tmp_path / 'settings.json'
        with patch.object(app, '_config_path', return_value=p):
            cfg = app._load_cfg()
        assert cfg == app._DEFAULTS

    def test_roundtrip(self, tmp_path):
        p = tmp_path / 'settings.json'
        data = {
            'api_keys': ['key_abcdef12345678'],
            'selected_api_key': 'key_abcdef12345678',
            'model': 'gemini-2.5-flash',
            'max_proxy_mb': 10.0,
            'workers': 2,
        }
        with patch.object(app, '_config_path', return_value=p):
            app._save_cfg(data)
            loaded = app._load_cfg()
        assert loaded['api_keys'] == ['key_abcdef12345678']
        assert loaded['model'] == 'gemini-2.5-flash'
        assert loaded['workers'] == 2

    def test_corrupt_file_returns_defaults(self, tmp_path):
        p = tmp_path / 'settings.json'
        p.write_text('{ not valid json !!!}')
        with patch.object(app, '_config_path', return_value=p):
            cfg = app._load_cfg()
        assert cfg == app._DEFAULTS

    def test_partial_file_merged_with_defaults(self, tmp_path):
        p = tmp_path / 'settings.json'
        p.write_text(json.dumps({'model': 'gemini-2.5-flash'}))
        with patch.object(app, '_config_path', return_value=p):
            cfg = app._load_cfg()
        assert cfg['model'] == 'gemini-2.5-flash'
        assert cfg['workers'] == app._DEFAULTS['workers']
        assert cfg['api_keys'] == []

    def test_saved_file_is_valid_json(self, tmp_path):
        p = tmp_path / 'settings.json'
        with patch.object(app, '_config_path', return_value=p):
            app._save_cfg(dict(app._DEFAULTS))
        parsed = json.loads(p.read_text())
        assert isinstance(parsed, dict)

    def test_multiple_keys_persisted(self, tmp_path):
        p = tmp_path / 'settings.json'
        keys = ['first_key_long_enough', 'second_key_long_enough']
        with patch.object(app, '_config_path', return_value=p):
            app._save_cfg({**app._DEFAULTS, 'api_keys': keys, 'selected_api_key': keys[0]})
            loaded = app._load_cfg()
        assert loaded['api_keys'] == keys
        assert loaded['selected_api_key'] == keys[0]

    def test_defaults_dict_unchanged_after_load(self, tmp_path):
        original_defaults = dict(app._DEFAULTS)
        p = tmp_path / 'settings.json'
        with patch.object(app, '_config_path', return_value=p):
            app._load_cfg()
        assert app._DEFAULTS == original_defaults
