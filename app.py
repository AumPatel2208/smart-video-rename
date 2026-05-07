#!/usr/bin/env python3
"""Smart Video Rename — desktop GUI. Run with: python app.py"""

from nicegui import ui, run
import video_rename


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


# ─── UI ───────────────────────────────────────────────────────────────────────

ui.label('Smart Video Rename').classes('text-2xl font-bold q-mb-sm')

with ui.tabs() as tabs:
    t_process = ui.tab('Process')
    t_restore = ui.tab('Backup / Restore')

with ui.tab_panels(tabs, value=t_process).classes('w-full'):

    # ── Process ───────────────────────────────────────────────────────────────
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
                value='gemma-3-27b-it', label='Model',
            ).classes('w-52')
            proxy_mb = ui.number('Max proxy MB', value=20.0, min=5, step=5).classes('w-28')
            workers_n = ui.number('Workers', value=4, min=1, max=16).classes('w-24')

        with ui.row().classes('items-center gap-6'):
            dry_run_cb = ui.checkbox('Dry run')
            slate_cb = ui.checkbox('Detect slate / clapperboard')
            parallel_cb = ui.checkbox('Parallel processing')

        api_key_in = ui.input(
            'Google API key',
            placeholder='Leave blank to use GOOGLE_API_KEY from env / .env file',
            password=True, password_toggle_button=True,
        ).classes('w-full')

        ui.separator()

        proc_log = ui.log(max_lines=500).classes('w-full h-64 font-mono text-xs')

        # Results table — hidden until a job succeeds
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

            proc_btn.set_enabled(False)
            results_card.visible = False
            proc_log.clear()

            try:
                result = await run.io_bound(
                    video_rename.run_processing,
                    directory,
                    api_key=api_key_in.value.strip() or None,
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

    # ── Backup / Restore ──────────────────────────────────────────────────────
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
