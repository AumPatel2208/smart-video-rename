#!/usr/bin/env python3
"""Smart Video Rename — desktop GUI. Run with: python app.py"""

import json
import os
import sys
from pathlib import Path

from nicegui import ui, run
import video_rename


# ── Config persistence ────────────────────────────────────────────────────────

def _config_path() -> Path:
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA', Path.home()))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    d = base / 'SmartVideoRename'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'settings.json'

_DEFAULTS: dict = {
    'api_keys': [],
    'selected_api_key': '',
    'model': 'gemma-3-27b-it',
    'max_proxy_mb': 20.0,
    'workers': 4,
}

def _load_cfg() -> dict:
    p = _config_path()
    if p.exists():
        try:
            return {**_DEFAULTS, **json.loads(p.read_text())}
        except Exception:
            pass
    return dict(_DEFAULTS)

def _save_cfg(cfg: dict) -> None:
    _config_path().write_text(json.dumps(cfg, indent=2))

def _mask(key: str) -> str:
    """Show only enough of a key to tell entries apart."""
    return key if len(key) <= 14 else f"{key[:8]}...{key[-4:]}"

def _key_options(keys: list) -> dict:
    """Build the options dict for ui.select. Empty list gets a placeholder entry."""
    if not keys:
        return {'': '— no saved keys —'}
    return {k: _mask(k) for k in keys}


# ── Folder / file pickers ─────────────────────────────────────────────────────

def _pick_folder(target: ui.input) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder = filedialog.askdirectory()
        root.destroy()
        if folder:
            target.set_value(folder)
    except Exception:
        ui.notify('Type the path directly (tkinter unavailable)', type='warning')


def _pick_file(target: ui.input) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        path = filedialog.askopenfilename(filetypes=[('JSON backup', '*.json')])
        root.destroy()
        if path:
            target.set_value(path)
    except Exception:
        ui.notify('Type the path directly (tkinter unavailable)', type='warning')


# ── UI ────────────────────────────────────────────────────────────────────────

cfg = _load_cfg()

ui.label('Smart Video Rename').classes('text-2xl font-bold q-mb-sm')

with ui.tabs() as tabs:
    t_process = ui.tab('Process')
    t_restore = ui.tab('Backup / Restore')

with ui.tab_panels(tabs, value=t_process).classes('w-full'):

    # ── Process tab ───────────────────────────────────────────────────────────
    with ui.tab_panel(t_process):

        with ui.row().classes('w-full items-end gap-2'):
            dir_in = ui.input('Video folder', placeholder='/path/to/videos').classes('flex-1')
            ui.button('Browse', on_click=lambda: _pick_folder(dir_in)).props('flat')

        with ui.row().classes('w-full items-end gap-2'):
            out_in = ui.input('Output folder', placeholder='blank = rename in-place').classes('flex-1')
            ui.button('Browse', on_click=lambda: _pick_folder(out_in)).props('flat')

        ui.separator()

        with ui.row().classes('items-end gap-4 flex-wrap'):
            model_sel = ui.select(
                ['gemma-3-27b-it', 'gemini-2.5-flash'],
                value=cfg['model'], label='Model',
                on_change=lambda e: _save_cfg({**cfg, 'model': e.value}),
            ).classes('w-52')
            proxy_mb = ui.number(
                'Max proxy MB', value=cfg['max_proxy_mb'], min=5, step=5,
                on_change=lambda e: _save_cfg({**cfg, 'max_proxy_mb': e.value}),
            ).classes('w-28')
            workers_n = ui.number(
                'Workers', value=cfg['workers'], min=1, max=16,
                on_change=lambda e: _save_cfg({**cfg, 'workers': int(e.value)}),
            ).classes('w-24')

        with ui.row().classes('items-center gap-6'):
            dry_run_cb = ui.checkbox('Dry run')
            slate_cb = ui.checkbox('Detect slate / clapperboard')
            parallel_cb = ui.checkbox('Parallel processing')

        # ── API key management ────────────────────────────────────────────────
        ui.label('Google API key').classes('text-sm font-medium q-mt-sm')

        initial_value = cfg['selected_api_key'] or (cfg['api_keys'][0] if cfg['api_keys'] else '')

        with ui.row().classes('w-full items-end gap-2'):
            key_select = ui.select(
                options=_key_options(cfg['api_keys']),
                value=initial_value,
                label='Saved keys',
                on_change=lambda e: _save_cfg({**cfg, 'selected_api_key': e.value}),
            ).classes('flex-1')

            def _remove_key() -> None:
                sel = key_select.value
                if not sel or sel not in cfg['api_keys']:
                    ui.notify('No key selected', type='warning')
                    return
                cfg['api_keys'].remove(sel)
                cfg['selected_api_key'] = cfg['api_keys'][0] if cfg['api_keys'] else ''
                key_select.set_options(_key_options(cfg['api_keys']), value=cfg['selected_api_key'])
                _save_cfg(cfg)
                ui.notify('Key removed', type='info')

            ui.button('Remove', on_click=_remove_key).props('flat color=negative')

        with ui.row().classes('w-full items-end gap-2'):
            new_key_in = ui.input(
                'Add new key', placeholder='Paste API key here',
                password=True, password_toggle_button=True,
            ).classes('flex-1')

            def _add_key() -> None:
                k = new_key_in.value.strip()
                if not k:
                    ui.notify('Enter a key first', type='warning')
                    return
                if k in cfg['api_keys']:
                    ui.notify('Key already saved', type='info')
                    key_select.set_options(_key_options(cfg['api_keys']), value=k)
                    return
                cfg['api_keys'].append(k)
                cfg['selected_api_key'] = k
                key_select.set_options(_key_options(cfg['api_keys']), value=k)
                new_key_in.set_value('')
                _save_cfg(cfg)
                ui.notify('Key saved', type='positive')

            ui.button('Add', on_click=_add_key).props('color=primary')

        ui.label(
            'No key selected? Falls back to GOOGLE_API_KEY in .env / environment.'
        ).classes('text-xs text-gray-400')

        ui.separator()

        proc_log = ui.log(max_lines=500).classes('w-full h-64 font-mono text-xs')

        results_card = ui.card().classes('w-full q-mt-sm')
        results_card.visible = False
        with results_card:
            ui.label('Results').classes('font-semibold q-mb-xs')
            results_tbl = ui.table(
                columns=[
                    {'name': 'orig', 'label': 'Original file', 'field': 'orig', 'align': 'left'},
                    {'name': 'new',  'label': 'Renamed to',    'field': 'new',  'align': 'left'},
                    {'name': 'desc', 'label': 'Description',   'field': 'desc', 'align': 'left'},
                    {'name': 'tags', 'label': 'Tags',          'field': 'tags', 'align': 'left'},
                ],
                rows=[],
            ).classes('w-full')

        async def start_processing() -> None:
            directory = dir_in.value.strip()
            if not directory:
                ui.notify('Select a video folder first', type='warning')
                return

            # Use the selected key (empty string → None → falls back to env var)
            api_key = key_select.value or None

            proc_btn.set_enabled(False)
            results_card.visible = False
            proc_log.clear()

            try:
                result = await run.io_bound(
                    video_rename.run_processing,
                    directory,
                    api_key=api_key,
                    dry_run=dry_run_cb.value,
                    output_dir=out_in.value.strip() or None,
                    parallel=parallel_cb.value,
                    workers=int(workers_n.value),
                    max_proxy_size=float(proxy_mb.value),
                    detect_slate=slate_cb.value,
                    model=model_sel.value,
                    on_progress=proc_log.push,
                )
                results_tbl.rows = [
                    {
                        'orig': e.original_filename,
                        'new':  e.new_filename,
                        'desc': e.new_metadata.description,
                        'tags': ', '.join(e.new_metadata.tags),
                    }
                    for e in result['entries']
                ]
                results_tbl.update()
                results_card.visible = bool(result['entries'])
                ui.notify(
                    f"Done — {result['processed']}/{result['total']} processed",
                    type='positive',
                )
            except Exception as exc:
                proc_log.push(f'\nERROR: {exc}')
                ui.notify(str(exc), type='negative')
            finally:
                proc_btn.set_enabled(True)

        proc_btn = ui.button('Process videos', on_click=start_processing).props('color=primary')

    # ── Backup / Restore tab ──────────────────────────────────────────────────
    with ui.tab_panel(t_restore):

        ui.label(
            'Restore videos to their original filenames and metadata using a backup JSON file.'
        ).classes('text-sm text-gray-500 q-mb-sm')

        with ui.row().classes('w-full items-end gap-2'):
            backup_in = ui.input('Backup JSON file').classes('flex-1')
            ui.button('Browse', on_click=lambda: _pick_file(backup_in)).props('flat')

        restore_dry_cb = ui.checkbox('Dry run (preview only)')

        restore_log = ui.log(max_lines=300).classes('w-full h-64 font-mono text-xs')

        async def start_restore() -> None:
            path = backup_in.value.strip()
            if not path:
                ui.notify('Select a backup file first', type='warning')
                return

            restore_btn.set_enabled(False)
            restore_log.clear()

            try:
                await run.io_bound(
                    video_rename.run_restore,
                    path,
                    dry_run=restore_dry_cb.value,
                    on_progress=restore_log.push,
                )
                ui.notify('Restore complete', type='positive')
            except Exception as exc:
                restore_log.push(f'\nERROR: {exc}')
                ui.notify(str(exc), type='negative')
            finally:
                restore_btn.set_enabled(True)

        restore_btn = ui.button('Restore', on_click=start_restore).props('color=warning')


ui.run(
    native=True,
    title='Smart Video Rename',
    window_size=(960, 720),
    reload=False,
)
